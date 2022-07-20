import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ummjl.util import *
# from resnet_model import resnet34
import math

# def gelu_new(x):
#     """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
#         Also see https://arxiv.org/abs/1606.08415
#     """
#     return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class BertRel(torch.nn.Module):
    def __init__(self, params, myvlbert):
        super(BertRel, self).__init__()
        self.params = params
        self.mybert = myvlbert
        self.senti_preject = torch.nn.Linear(in_features=768, out_features=2)
        if self.params.class_name == "it":
            self.pair_preject = torch.nn.Linear(in_features=768, out_features=4)
        else:
            self.pair_preject = torch.nn.Linear(in_features=768, out_features=2)


    def forward(self, text, img, lens, masks, relation=None,mode="train",indexs_list=None):

        all_encoder_layers, attention_probs = self.mybert(img, text,relation=relation)
        if relation is not None:
            pair_out = self.pair_preject(all_encoder_layers[:, :1, :])
        else:
            pair_out = self.senti_preject(all_encoder_layers[:, :1, :])

        return pair_out

