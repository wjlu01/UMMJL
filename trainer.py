from torch import threshold
import torch.utils.data
from ummjl.bert_rel import *
from ummjl.bert_rel import BertRel
from timeit import default_timer as timer
from util import *
from tqdm import tqdm
import numpy as np
from flair.embeddings import *
import gc
from transformers import get_linear_schedule_with_warmup
import os


def burnin_schedule(i):
    if i < 10:
        factor = 1
    elif i < 20:
        factor = 0.1
    else:
        factor = 0.01
    return factor


class Trainer:
    def __init__(self, params, config, data_loader, data_loader_bb, evaluator, pre_model=None):
        self.params = params
        self.config = config
        self.auxiliary_data_loader = data_loader
        self.main_data_loader = data_loader_bb
        self.evaluator = evaluator
        self.pre_model = pre_model

    def train(self):
        embedding_types = [
            WordEmbeddings(
                'pretrained/embeddings/en-fasttext-crawl-300d-1M'),
            CharacterEmbeddings('pretrained/embeddings/common_characters_large'),
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        ner_model = BertRel(self.params, self.pre_model)

        loss_function_relation = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            ner_model = ner_model.cuda()
            loss_function_relation = loss_function_relation.cuda()

        paras = dict(ner_model.named_parameters())
        paras_new = []
        for k, v in paras.items():
            if 'pre_resnet' in k or 'vlbert' in k:
                paras_new += [{'params': [v], 'lr':self.params.lr_1}] #*
            else:
                paras_new += [{'params': [v], 'lr':self.params.lr_2}] #*

        optimizer = torch.optim.AdamW(paras_new, weight_decay=self.params.wdecay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

        if self.params.class_name=="txt":
            num_epochs=15
        elif self.params.class_name=="it":
            num_epochs=20
            if self.params.task_name=="fake_news":
                num_epochs=25
        else:
            num_epochs=10
        
        if not os.path.exists(self.params.model_dir):
            print(self.params.model_dir)
            os.mkdir(self.params.model_dir)
    

        try:
            prev_best = 0
            best_epoch = 0
            for epoch in range(num_epochs):
                losses = []
                start_time = timer()

############################################## Training Auxiliary Task #########################################
                if not self.params.hard_mask:
                    for (x, _, img, sent, mask, lens,targets,indexs_list)  in tqdm(self.auxiliary_data_loader.train_data_loader):
                        ner_model.train()
                        optimizer.zero_grad()
                        preds = ner_model(to_variable(sent), to_variable(img),lens, to_variable(mask))
                        targets = to_variable(targets).contiguous()
                        # computing CrossEntropyLoss
                        loss = loss_function_relation(preds.squeeze(1), targets)

                        loss.backward()
                        optimizer.step()
                    gc.collect()
                    torch.cuda.empty_cache()

#################################### Training Main Task (Relation Inference) ####################################

                for (x, _, img, sent, mask, lens,targets,indexs_list) in tqdm(self.main_data_loader.train_data_loader):
                    ner_model.train()
                    # compute relation mask
                    if not self.params.hard_mask:
                        with torch.no_grad():
                            relation = ner_model(to_variable(sent), to_variable(img),lens, to_variable(mask))
                            relation = F.softmax(relation,dim=-1)
                    else:
                        relation=torch.zeros((self.params.batch_size,1),device=self.params.cuda)
                                
                    optimizer.zero_grad()

                    # predict relation
                    preds= ner_model(to_variable(sent), to_variable(img),lens, to_variable(mask),relation=relation.detach())  # seq_len * bs * labels
                    targets = to_variable(targets).contiguous()
                    # computing CrossEntropyLoss
                    loss1 = loss_function_relation(preds.squeeze(1), targets)

                    loss1.backward()
                    losses.append(loss1.data.cpu().numpy())
                    optimizer.step()

                scheduler.step()
                optim_state = optimizer.state_dict()

                gc.collect()        
                torch.cuda.empty_cache()
                # Calculate wei-F1 and save best model
                if (epoch + 1) % self.params.validate_every == 0:

                    with torch.no_grad():
                        dev_loss,acc_dev, f1_dev, p_dev, r_dev = self.evaluator.get_weightedF1(ner_model, 'val',loss_function_relation)

                        print(
                            "Epoch {} : Training Loss: {:.5f}\n,dev loss:{:.5f} Acc: {:.5f}, F1: {:.5f}, Prec: {:.5f}, Rec: {:.5f}, LR: {:.5f}"
                            "Time elapsed {:.2f} mins"
                                .format(epoch + 1, np.asscalar(np.mean(losses)), 
                                        np.asscalar(np.mean(dev_loss)),
                                        acc_dev, f1_dev, p_dev, r_dev,
                                        optim_state['param_groups'][0]['lr'],
                                        (timer() - start_time) / 60))

                        if f1_dev > prev_best:
                            tmp=f1_dev
                            print("f1-score increased....saving weights !!")
                            best_epoch = epoch + 1
                            prev_best=tmp
                            model_path = self.params.model_dir + "/epoch{}_f1_{:.5f}.pth".format(epoch + 1, tmp)
                            torch.save(ner_model.state_dict(), model_path)
                            print("UMMJL save in " + model_path)
                            
                else:
                    print("Epoch {} : Training Loss: {:.5f}".format(epoch + 1, np.asscalar(np.mean(losses))))
                torch.cuda.empty_cache()
                
                if epoch + 1 == num_epochs:
                    best_model_path = self.params.model_dir + "/epoch{}_f1_{:.5f}.pth".format(best_epoch, prev_best)
                    print("{} epoch get the best f1 {:.5f}".format(best_epoch, prev_best))
                    print("the UMMJL is save in " + best_model_path)
                    
                    model_file_path = best_model_path
                    torch.cuda.empty_cache()
                    ner_model.load_state_dict(torch.load(model_file_path))

                    print("Evaluating UMMJL on test set...")
                    loss,acc, f1, prec, rec = self.evaluator.get_weightedF1(ner_model, 'test')
                    print("Accuracy : {}".format(acc))
                    print("F1 : {}".format(f1))
                    print("Precision : {}".format(prec))
                    print("Recall : {}".format(rec))
                    print("Evaluating UMMJL on test set...[OK]")
                    
        except KeyboardInterrupt:
            print("Interrupted..")
