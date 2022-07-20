import os
import argparse

from torch.nn import parameter
from ummjl.bert_rel import *
from data_loader_auxiliary import DataLoader
from data_loader_bb import DataLoader as DLbb
from evaluator import Evaluator
from trainer import Trainer
import random
import numpy as np
import torch
import flair
from cfgs.config import config, update_config
from ummjl.resnet_vlbert import ResNetVLBERT
from flair.embeddings import *

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for UMMJL')
    parser.add_argument("--cuda", dest="cuda", type=int, default=5) 
    parser.add_argument("--main_task_img_dir", dest="main_task_img_dir", type=str,
                        default='datasets/bloomberg/rel_img/') 
    parser.add_argument("--main_task_split_file", dest="main_task_split_file", type=str, default='datasets/bloomberg/')
 #pre_split_file

    parser.add_argument("--auxiliary_task_split_file", dest="auxiliary_task_split_file", type=str, 
                        default='datasets/caption/') 
    parser.add_argument("--auxiliary_task_img_dir", dest="auxiliary_task_img_dir", type=str,
                        default='datasets/caption/img/') 
    parser.add_argument("--task_name",dest="task_name",type=str,default="caption") 
    parser.add_argument("--hard_mask",dest="hard_mask",type=int,default=0) 
    parser.add_argument("--hard_mask_value",dest="hard_mask_value",type=float,default=1) 

    parser.add_argument("--class_name",dest="class_name",default="it",choices=["img", "txt", "it"]) 
    parser.add_argument("--seed",dest="seed",type=int,default=0) 

    parser.add_argument("--dl_train",dest="dl_train",type=str,default="train") 
    parser.add_argument("--lr_1", dest="lr_1", type=float, default=1e-5) 
    parser.add_argument("--lr_2", dest="lr_2", type=float, default=1e-5) 
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.5)
    parser.add_argument("--mask_part",dest="mask_part",type=str,default="all_1") 
    parser.add_argument("--pretrain_load", dest="pretrain_load", type=int, default=1)
    parser.add_argument("--pre_hidden_dimension", dest="pre_hidden_dimension", type=int, default=256)
    parser.add_argument("--cat_h_e", dest="cat_h_e", type=int, default=1)

    MODEL_DIR = 'save_model/'  
    
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=8)
    parser.add_argument("--mode", dest="mode", type=int, default=0) 
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_file_name", dest="model_file_name", type=str, default="11")
    
    parser.add_argument("--hidden_size", dest="hidden_size", type=int, default=768)    
    parser.add_argument("--hidden_dimension", dest="hidden_dimension", type=int, default=512)
    parser.add_argument("--lr", dest="lr", type=float, default=5e-5) 
    parser.add_argument("--n_layers", dest="n_layers", type=int, default=3)
    parser.add_argument("--clip_value", dest="clip_value", type=float, default=5)
    parser.add_argument("--wdecay", dest="wdecay", type=float, default=0.0000001)
    parser.add_argument("--step_size", dest="step_size", type=int, default=15)
    parser.add_argument("--gamma", dest="gamma", type=float, default=0.01)
    parser.add_argument("--validate_every", dest="validate_every", type=int, default=1)
    parser.add_argument("--sent_maxlen", dest="sent_maxlen", type=int, default=35)
    parser.add_argument("--word_maxlen", dest="word_maxlen", type=int, default=41)

    parser.add_argument('--cfg', type=str, help='path to config file',
                        default='cfgs/base_gt_boxes_4x16G.yaml')
    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)

    return args, config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    params, config = parse_arguments()
    set_seed(params.seed)
    print("Constructing data loaders...")

    device_id =params.cuda
    torch.cuda.set_device(device_id)
    flair.device = torch.device('cuda:%d' % device_id)


    if params.pretrain_load == 1:

        myvlbert = ResNetVLBERT(config,params)
        
        pretrained_bert_model = torch.load('pretrained/bert-base-uncased/pytorch_model.bin')
        new_state_dict = myvlbert.state_dict()
        miss_keys = []
        for k in new_state_dict.keys():
            key = k.replace('vlbert', 'bert') \
                .replace('LayerNorm.weight', 'LayerNorm.gamma') \
                .replace('LayerNorm.bias', 'LayerNorm.beta')
            if key in pretrained_bert_model.keys():
                new_state_dict[k] = pretrained_bert_model[key]
            else:
                miss_keys.append(k)

        myvlbert.load_state_dict(new_state_dict)
        print('Load pretrain UMMJL...[OK]')

    dl = DataLoader(params)
    dlbb = DLbb(params) 
    evaluator = Evaluator(params,dl, dlbb) 
    print("Constructing data loaders...[OK]")

    if params.mode == 0: 
        print("Training...")
        t = Trainer(params, config, dl, dlbb, evaluator, myvlbert) 
        t.train() 
        print("Training...[OK]")
    elif params.mode == 1: 
        embedding_types = [
            WordEmbeddings('pretrained/embeddings/en-fasttext-crawl-300d-1M'),
            CharacterEmbeddings('pretrained/embeddings/common_characters_large'),
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types) 

        print("Loading UMMJL...")
        model=BertRel(params,myvlbert)
        model_file_path = os.path.join(params.model_dir, params.model_file_name)
        torch.cuda.empty_cache()
        model.load_state_dict(torch.load(model_file_path))
        if torch.cuda.is_available():
            model = model.cuda()
        print("Loading UMMJL...[OK]")

        print("Evaluating UMMJL on test set...")
        _,acc, f1, prec, rec = evaluator.get_weightedF1(model, 'test')
        print("Accuracy : {}".format(acc))
        print("F1 : {}".format(f1))
        print("Precision : {}".format(prec))
        print("Recall : {}".format(rec))
        print("Evaluating UMMJL on test set...[OK]")


if __name__ == '__main__':
    main()
