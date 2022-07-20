from util import *
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import math 
from torch import nn

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Evaluator:
    def __init__(self, params, data_loader,data_loader_bb):
        self.params = params
        self.auxiliary_data_loader = data_loader
        self.main_data_loader = data_loader_bb

    def get_weightedF1(self, model, split, loss_function_relation=nn.CrossEntropyLoss().cuda()):
        if split == 'val':
            data_loader = self.main_data_loader.val_data_loader
        else:
            data_loader = self.main_data_loader.test_data_loader

        model.eval()
        labels_pred = []
        labels = []
        relations=[]
        losses=[]
        with torch.no_grad():
            # for (x, _, img, sent, mask, lens,targets,indexs_list) in tqdm(data_loader):
            for (x, _, img, sent, mask, lens, targets, indexs_list) in data_loader:
                # compute relation mask
                relation = model(to_variable(sent), to_variable(img), mask, lens, mode="test")  # seq_len * bs * labels
                relation = F.softmax(relation,dim=-1)

                idx = int(self.params.mask_part.strip().split("_")[1])

                relations.append(relation.cpu().numpy()[0][0][idx])

                probs = model(to_variable(sent), to_variable(img), mask, lens, mode="test", relation=relation.detach())  # seq_len * bs * labels

                relation_out=torch.argmax(probs.squeeze(1),axis=1)
                loss=loss_function_relation(probs.squeeze(1), to_variable(targets).contiguous())
                losses.append(loss.data.cpu().numpy())

                labels.append(int(to_variable(targets).contiguous().cpu().numpy()[0]))
                labels_pred.append(int(relation_out.cpu().numpy()[0]))

        return self.evaluate(labels_pred, labels, losses)

    def evaluate(self, labels_pred, labels, losses):

        wei_f1 = f1_score(labels,labels_pred,average='weighted')
        acc = accuracy_score(labels, labels_pred)
        macro_f1 = f1_score(labels,labels_pred,average='macro')
        macro_r=recall_score(labels, labels_pred, average='macro')
        macro_p=precision_score(labels, labels_pred, average='macro')
        print("macro f1",macro_f1, "macro recall", macro_r, "macro prec", macro_p)

        return losses,acc, wei_f1, macro_p, macro_r

