import traceback
from numbers import Number
from pytorch_pretrained_bert import BertTokenizer
from util import *
import torch.utils.data
import os
from collections import Counter
import numpy as np
import gensim
from gensim.models import word2vec
from gensim.models import fasttext
import  torch
import random
from flair.data import Sentence
import torchvision.transforms as transforms
from PIL import Image
import csv
import six
import pickle
import lmdb
import os.path as osp

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.48, 0.498, 0.531),
                         (0.214, 0.207, 0.207))]
)
def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self, params, x, x_flair, y, img_id, s_idx, e_idx,ifparis,indexs_list):
        self.params = params
        self.x = x
        self.x_flair = x_flair
        self.y = y
        self.img_id  = img_id
        self.s_idx = s_idx
        self.e_idx = e_idx
        self.num_of_samples = e_idx - s_idx
        self.ifpairs = ifparis
        self.indexs_list=indexs_list

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        x = self.x[self.s_idx + idx]
        x_flair = self.x_flair[self.s_idx + idx]
        y = self.y[self.s_idx + idx]
        img_id = self.img_id[self.s_idx + idx]
        indexs_list=self.indexs_list[self.s_idx + idx]

        path = os.path.join(self.params.auxiliary_task_img_dir, img_id + '.jpg')
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        obj_x = np.array(image)
        ifpairs = self.ifpairs[self.s_idx + idx]
        return x, x_flair, y, obj_x,ifpairs,indexs_list

    def collate(self, batch):  #x是mask的token y是tokened的token indexs_list与y对齐
        x = np.array([x[0] for x in batch])
        x_flair = [x[1] for x in batch]
        y = np.array([x[2] for x in batch])
        obj_x = np.array([x[3] for x in batch])
        ifpairs = np.array([x[-2] for x in batch])
        indexs_list=np.array([x[-1] for x in batch])

        bool_mask = y == 0

        mask = 1 - bool_mask.astype(np.int)

        # index of first 0 in each row, if no zero then idx = -1
        zero_indices = np.where(bool_mask.any(1), bool_mask.argmax(1), -1).astype(np.int)
        input_len = np.zeros(len(batch))
        for i in range(len(batch)):
            if zero_indices[i] == -1:
                input_len[i] = len(y[i])
            else:
                input_len[i] = zero_indices[i]
        sorted_input_arg = np.argsort(-input_len)

        x = x[sorted_input_arg]
        x_flair = sorted(x_flair, key=lambda i: len(i), reverse=True)
        ifpairs = ifpairs[sorted_input_arg]
        y = y[sorted_input_arg]
        indexs_list=indexs_list[sorted_input_arg]

        obj_x = obj_x[sorted_input_arg]
        mask = mask[sorted_input_arg]

        input_len = input_len[sorted_input_arg]

        max_seq_len = int(input_len[0])

        trunc_x = np.zeros((len(batch), max_seq_len))
        trunc_x_flair = []
        trunc_indexs_list=[]

        trunc_y = np.zeros((len(batch), max_seq_len))
        trunc_mask = np.zeros((len(batch), max_seq_len))
        for i in range(len(batch)):

            trunc_x_flair.append(x_flair[i])
            trunc_x[i,:min(max_seq_len,len(x[i]))] = x[i, :min(max_seq_len,len(x[i]))]

            trunc_y[i] = y[i, :max_seq_len]
            trunc_mask[i,:min(max_seq_len,len(mask[i]))] = mask[i, :min(max_seq_len,len(mask[i]))]
            trunc_indexs_list.append(indexs_list[i])

        return to_tensor(trunc_x).long(), trunc_x_flair, to_tensor(obj_x), to_tensor(trunc_y).long(), to_tensor(trunc_mask).long(), \
               to_tensor(input_len).int(),to_tensor(ifpairs).long(),trunc_indexs_list


class DataLoader:
    def __init__(self, params):
        '''
        self.x : sentence encoding with padding at word level
        self.y : label corresponding to the words in the sentences
        :param params:
        '''
        self.params = params

        self.sentences, self.datasplit, \
            self.x, self.x_flair, self.y, \
            self.num_sentence, self.img_id,self.ifpairs,self.indexs_list\
            = self.load_data()

        kwargs = {'num_workers': 12, 'pin_memory': True} if torch.cuda.is_available() else {}

        dataset_train = CustomDataSet(params, self.x,  self.x_flair, self.y, self.img_id, self.datasplit[0], self.datasplit[1],self.ifpairs,self.indexs_list)
        self.train_data_loader = torch.utils.data.DataLoader(dataset_train,
                                                             batch_size=self.params.batch_size,
                                                             collate_fn=dataset_train.collate,
                                                             shuffle=True, **kwargs)

        dataset_test = CustomDataSet(params, self.x,  self.x_flair, self.y, self.img_id, self.datasplit[1], self.datasplit[2],self.ifpairs,self.indexs_list)
        self.test_data_loader = torch.utils.data.DataLoader(dataset_test,
                                                            batch_size=1,
                                                            collate_fn=dataset_test.collate,
                                                            shuffle=False, **kwargs)

    def load_data(self):
        print('calculating vocabulary...')

        datasplit, sentences, sent_maxlen, word_maxlen, num_sentence,  img_id ,ifpairs= self.load_sentence(
            'IMGID', self.params.auxiliary_task_split_file, self.params.dl_train,'test')

        x, x_flair, y,indexs_list = self.pad_sequence(sentences,
                                               word_maxlen=self.params.word_maxlen, sent_maxlen=sent_maxlen)

        return [sentences, datasplit, x, x_flair, y, num_sentence,
                img_id,ifpairs,indexs_list]

    def load_sentence(self, IMAGEID, tweet_data_dir, train_name,test_name):
        """
        read the word from doc, and build sentence. every line contain a word and it's tag
        every sentence is split with a empty line. every sentence begain with an "IMGID:num"

        """
        # IMAGEID='IMGID'
        img_id = []
        sentences = []
        sentence = []
        sent_maxlen = 0
        word_maxlen = 0
        datasplit = []
        ifpairs =  []

        import ast
        import csv
        import string

        def datafilter(token):
            if token.startswith("http") or token=="RT" or token in string.punctuation:
                return False
            else:
                return True

        for fname in (train_name,test_name):
            datasplit.append(len(img_id))

            if self.params.task_name=="fake_news":
                with open(os.path.join(tweet_data_dir, fname),'r', encoding='utf-8') as file:
                    lines=file.readlines()
                    for line in lines:
                        line=line.strip().split("\t")
                        num = line[0][:-4]
                        img_id.append(num)
                        sentence=line[1].strip().split(" ")[:30]
                        sentences.append(sentence)
                        sent_maxlen = max(sent_maxlen, len(sentence))
                        for word in sentence:
                            word_maxlen = max(word_maxlen, len(word))
                        ifpairs.append(int(line[-1]))

            elif self.params.task_name=="caption":
                with open(os.path.join(tweet_data_dir, fname),'r', encoding='utf-8') as file:
                    lines=file.readlines()
                    for line in lines:
                        line=line.strip().split("\t")
                        num = line[0][:-4]
                        img_id.append(num)
                        sentence=line[1].split(' ')
                        sentences.append(sentence)
                        sent_maxlen = max(sent_maxlen, len(sentence))
                        for word in sentence:
                            word_maxlen = max(word_maxlen, len(word))
                        ifpairs.append(int(line[-1]))
            
            elif self.params.task_name=="senti":
                with open(os.path.join(tweet_data_dir, fname), 'r', encoding='utf-8') as file:
                    reader=file.readlines()
                    for line in reader:
                        line=line.split("\t")
                        num = line[0]
                        img_id.append(num)
                        sentence=line[1].split()
                        sentence=[word for word in filter(datafilter,sentence)]
                        sentences.append(sentence)
                        sent_maxlen = max(sent_maxlen, len(sentence))
                        for word in sentence:
                            word_maxlen = max(word_maxlen, len(word))
                        #image-text task
                        ifpairs.append(int(line[-3]))


        datasplit.append(len(img_id))
        num_sentence = len(sentences)

        print("datasplit", datasplit)
        print("dlbb sentence",sentences[len(sentences) - 2])
        print("dlbb sentence",sentences[0])
        print('sent_maxlen', sent_maxlen)
        print('word_maxlen', word_maxlen)
        print('number sentence', len(sentences))
        print('number image', len(img_id))
        return [datasplit, sentences, sent_maxlen, word_maxlen, num_sentence, img_id,ifpairs]

    @staticmethod
    def label_index(labels_counts):
        """
           the input is the output of Counter. This function defines the (label, index) pair,
           and it cast our datasets label to the definition (label, index) pair.
        """

        num_labels = len(labels_counts)
        labelVoc_inv = [x[0] for x in labels_counts.most_common()]

        labelVoc = {'0': 0,
                    'B-PER': 1, 'I-PER': 2,
                    'B-LOC': 3, 'I-LOC': 4,
                    'B-ORG': 5, 'I-ORG': 6,
                    'B-OTHER': 7, 'I-OTHER': 8,
                    'O': 9}
        if len(labelVoc) < num_labels:
            for key, value in labels_counts.items():
                if not labelVoc.has_key(key):
                    labelVoc.setdefault(key, len(labelVoc))
        return labelVoc_inv, labelVoc

    @staticmethod
    def pad_sequences(y, sent_maxlen):
        padded = np.zeros((len(y), sent_maxlen))
        for i, each in enumerate(y):
            trunc_len = min(sent_maxlen, len(each))
            padded[i, :trunc_len] = each[:trunc_len]
        return padded.astype(np.int32)

    def pad_sequence(self, sentences, word_maxlen=40,
                     sent_maxlen=45):
        """
            This function is used to pad the word into the same length, the word length is set to 30.
            Moreover, it also pad each sentence into the same length, the length is set to 35.

        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        x = []
        x_flair = []
        y = []
        
        for sentence in sentences:
            w_id = []
            st = Sentence()
            for idx, word_label in enumerate(sentence):
                try:
                    w_id.append(tokenizer.vocab[word_label.lower()])
                except Exception as e:
                    w_id.append(tokenizer.vocab['[MASK]'])
                st.add_token(word_label)
            x.append(w_id)
            x_flair.append(st)


        x = self.pad_sequences(x, sent_maxlen)
        x = np.asarray(x)
        indexs_list=[]

        for sentence in sentences:
            w_id = []

            for word_label in sentence:
                w_id.append(word_label)
            w_id = w_id[:] 

            bert_tokenization = []
            index_list=[]
            index_list.append(-1)

            for i,token in enumerate(w_id):
                subtokens = tokenizer.tokenize(token)
                bert_tokenization.extend(subtokens)
                index_list.extend([i]*len(subtokens))
            if len(bert_tokenization) > sent_maxlen:
                sent_maxlen = len(bert_tokenization)
            index_list.append(-1)

            tokens = list()
            tokens.append("[CLS]")
            for token in bert_tokenization:
                tokens.append(token)
            tokens.append("[SEP]")

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            y.append(input_ids[:])
            indexs_list.append(index_list)

        sent_maxlen += 2
        y = self.pad_sequences(y, sent_maxlen)
        y = np.asarray(y)
        print("dlbb y",y,y.shape)
        indexs_list=self.pad_sequences(indexs_list,sent_maxlen)
        return [x, x_flair, y,indexs_list]
