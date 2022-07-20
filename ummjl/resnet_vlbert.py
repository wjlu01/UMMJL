import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from ummjl.module import Module
from ummjl.visual_linguistic_bert import VisualLinguisticBert
BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERT(Module):
    def __init__(self, config,params):
        super(ResNetVLBERT, self).__init__(config)
        self.params=params
        self.config = config
        
        # load pretrained ResNet-152
        resnet = torchvision.models.resnet152(pretrained=False)
        resnet.load_state_dict(torch.load('pretrained/resnet/resnet152-b121ed2d.pth'))
        modules = list(resnet.children())[:-2]
        self.pre_resnet = nn.Sequential(*modules)
        print('load resnet152 pretrained UMMJL')
        
        self.object_visual_embeddings = nn.Linear(2048, self.params.hidden_size)
        self.object_linguistic_embeddings = nn.Embedding(1, self.params.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN
        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)
        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,params)

        # init weights
        self.init_weight()

    def init_weight(self):
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)

    def fix_params(self):
        for param in self.image_feature_extractor.parameters():
            param.requires_grad = False
        for param in self.vlbert.parameters():
            param.requires_grad = False

    def forward(self,
                image,
                expression,
                relation=None,indexs_list=None):
        
        batch_size = expression.shape[0]
        dv=expression.device
        
        # load image
        images = image
        img_feature = self.pre_resnet(images)
        img_feature = img_feature.view(batch_size, 2048, 7 * 7).transpose(2, 1)
        box_mask = torch.ones((batch_size, 49), dtype=torch.bool, device=dv)
        object_visual_embedding = self.object_visual_embeddings(img_feature)

        # load text
        text_input_ids = expression
        text_token_type_ids = text_input_ids.new_zeros(text_input_ids.shape)
        text_mask = text_input_ids > 0
        text_visual_embeddings = torch.zeros(
            (text_input_ids.shape[0], text_input_ids.shape[1], self.params.hidden_size),
            device=dv,
        )
        object_linguistic_embeddings = self.object_linguistic_embeddings(
            box_mask.new_zeros((box_mask.shape[0], box_mask.shape[1])).long()
        )
        
        # concat image & text
        object_vl_embeddings = torch.cat((object_visual_embedding, object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states_text, hidden_states_image, _, attention_probs = self.vlbert(text_input_ids,
                                                                   text_token_type_ids,
                                                                   text_visual_embeddings,
                                                                   text_mask,
                                                                   object_vl_embeddings,
                                                                   box_mask,
                                                                   output_all_encoded_layers=False,
                                                                   output_text_and_object_separately=True,
                                                                   relation=relation,indexs_list=indexs_list)
        return hidden_states_text, attention_probs
