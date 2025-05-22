# Angular deep supervised hashing for image retrieval
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
from functions.loss.base_cls import BaseClassificationLoss
from transformers.models.bert.configuration_bert import BertConfig
from torch import Tensor, device, dtype, nn
from transformers.activations import ACT2FN

import random


from models.architectures.xbert import BertConfig, BertModel, BertOnlyMLMHead, ACT2FN



class Replace_Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        replace_kind_num = 2
        self.dense_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.act_fun = ACT2FN[config.hidden_act]
        self.norm_layer = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.predict_layer = nn.Linear(config.hidden_size, replace_kind_num)

    def forward(self, hidden_embedding):
        hidden_feat = self.dense_layer(hidden_embedding)
        hidden_feat = self.act_fun(hidden_feat)
        hidden_feat = self.norm_layer(hidden_feat)
        hidden_feat = self.predict_layer(hidden_feat)

        return hidden_feat
    
    
class PIXELLoss(BaseClassificationLoss):
    def __init__(self, alpha=1, beta=1, s=10, m=0.2, multiclass=False, method='cosface', **kwargs):
        super().__init__()
        logging.info("Loss parameters are: ", locals())
        self.alpha = alpha
        self.beta = beta
        self.s = s
        self.m = m

        self.method = method
        self.multiclass = multiclass
        self.weight = None
       
        self.ds = kwargs['dataset']

        if kwargs['dataset']=='cub':
            nclass=200 
            nattr = 312 
            nattr_embed = 512 
            max_words = 180
            self.w_tr = 0.5
            self.w_se = 2.5
        elif kwargs['dataset']=='awa2':
            nclass=50
            nattr = 85 
            nattr_embed = 512 
            max_words = 180
            self.w_tr = 0.4 
            self.w_se = 3   
        elif kwargs['dataset']=='sun':
            nclass=717
            nattr = 102 
            nattr_embed = 512 
            max_words = 180 
            self.w_tr = 0.3
            self.w_se = 1



        self.weight_ce = torch.nn.Parameter(torch.eye(nclass), requires_grad=False)
        self.is_bias = True
        self.is_conservative = True
        self.log_softmax_func = torch.nn.LogSoftmax(dim=1)



        self.bert_config = BertConfig.from_json_file("./bert/bert_config.json")   
        self.text_encoder = BertModel(config=self.bert_config, add_pooling_layer=False) 
        text_width = self.text_encoder.config.hidden_size
        embed_dim = 8
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.combine_text_proj = nn.Linear(embed_dim, embed_dim)
        # self.text_encoder_m = BertModel(config=self.bert_config, add_pooling_layer=False)     
        self.decoder_layer = BertOnlyMLMHead(config=self.bert_config)
        self.replace_predict_layer = Replace_Predictor(config=self.bert_config)

        
        self.attr_text_proj = nn.Linear(max_words, nattr)
        self.attr_embed_proj = nn.Linear(768, nattr_embed)


        self.alpha = torch.nn.Parameter(torch.Tensor([1]), requires_grad=True)


    def get_hamming_distance(self):
        assert self.weight is not None, 'please set weights before calling this function'

        b = torch.div(self.weight, self.weight.norm(p=2, dim=-1, keepdim=True) + 1e-7)
        b = torch.tanh(b)

        nbit = b.size(1)

        hd = 0.5 * (nbit - b @ b.t())
        return hd

    def get_margin_logits(self, logits, labels):
        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)

        if self.method == 'arcface':
            arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
            logits = torch.cos(arc_logits + y_onehot)
            margin_logits = self.s * logits
        else:
            y_onehot = torch.zeros_like(logits)
            y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
            margin_logits = self.s * (logits - y_onehot)

        return margin_logits

    def get_dynamic_logits(self, logits, code_logits, labels):
        """
        return dynamic generated labels
        :param logits:
        :param code_logits:
        :param labels:
        :return:
        """
        assert self.weight is not None
        new_weight = []
        new_labels = F.one_hot(torch.tensor([labels.size(1)] * labels.size(0)), labels.size(1) + 1).to(labels.device)
        for l in range(labels.size(0)):
            w = self.weight[labels[l].bool()].mean(dim=0, keepdim=True)  # (L,D)
            new_weight.append(w)
        new_weight = torch.cat(new_weight, dim=0)  # (BS, D)

        code_logits_norm = torch.div(code_logits, code_logits.norm(p=2, dim=1, keepdim=True) + 1e-7)
        new_weight_norm = torch.div(new_weight, new_weight.norm(p=2, dim=1, keepdim=True) + 1e-7)
        new_logits = (code_logits_norm * new_weight_norm).sum(dim=1, keepdim=True)  # (BS, D) * (BS, D) -> (BS, 1)
        new_logits = torch.cat([logits, new_logits], dim=1)
        return new_logits, new_labels



    def compute_loss_semantics_dist(self, img_attr_embeds, text_input_ids, text_attention_mask, labels):
        """
        img_attr_embed:  bs,attr_num,attr_embed_dim  e.g.  12,312,128 for cub

        """
        bs,attr_num,embed_dim = img_attr_embeds.shape
        img_attr_embeds = F.normalize(img_attr_embeds, dim=-1)

        # fashion text embedding
        mask_token_id = 103    
        text_dropout = 0.1

        for i in range(len(text_input_ids)):
            for j in range(len(text_input_ids[i])):
                if text_input_ids[i][j]!=mask_token_id:
                    if (random.random() < text_dropout):
                        text_input_ids[i][j] = mask_token_id

        text_input_ids = text_input_ids.cuda()
        text_attention_mask = text_attention_mask.cuda()

        text_output = self.text_encoder(text_input_ids, attention_mask = text_attention_mask,                      
                                        return_dict = True, mode = 'text')
                
        text_embeds = text_output.last_hidden_state

        text_attr_embeds = self.attr_embed_proj(text_embeds)
        text_attr_embeds = text_attr_embeds.permute(0,2,1)
        text_attr_embeds = self.attr_text_proj(text_attr_embeds)
        text_attr_embeds = text_attr_embeds.permute(0,2,1)

        text_attr_embeds = F.normalize(text_attr_embeds, dim=-1)
    
        loss = 0
        count = 0
        for i in range(len(labels)-1):
            for j in range(i+1, len(labels)):
                img_dist = img_attr_embeds[i] - img_attr_embeds[j]
                text_dist = text_attr_embeds[i] - text_attr_embeds[j]
                # img_dist = F.cosine_similarity(img_attr_embeds[i], img_attr_embeds[j], dim=-1)
                # text_dist = F.cosine_similarity(text_attr_embeds[i], text_attr_embeds[j], dim=-1)
                loss_one = torch.mean(torch.pow(img_dist - text_dist*self.alpha[0],2))#
                if labels[i]==labels[j]:
                    loss += (torch.mean(img_dist)+torch.mean(text_dist))*4

                loss += loss_one
                count+=1

        loss/=count  
  
        return loss


    def forward(self, logits, code_logits, labels, 
        attr_data, img_attr_embed,
        text_input_ids, text_attention_mask, text_mask_labels, text_replace_labels,
        package=None,
        onehot=True):
        
        if self.multiclass:
       
            logits, labels = self.get_dynamic_logits(logits, code_logits, labels)

            labels = labels.argmax(1)
        else:
            if onehot:
                labels = labels.argmax(1)


        embed = package['my_img_embed']
        att = package['att']
        if len(labels.size()) == 1:
            labels_onehot = self.weight_ce[labels]
        # tgt = torch.matmul(package['batch_label'], att)
        tgt = torch.matmul(labels_onehot, att)
        loss_reg = F.mse_loss(embed, tgt, reduction='mean')

        tr_loss = loss_reg * self.w_tr


        margin_logits = self.get_margin_logits(logits, labels)
        ce = F.cross_entropy(margin_logits, labels)


        semantics_dist_loss = self.compute_loss_semantics_dist(img_attr_embed, text_input_ids, text_attention_mask, labels)
        
        semantics_dist_loss *= self.w_se

        self.losses['ce'] = ce

        loss = ce + semantics_dist_loss + tr_loss 

        return loss
