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
       

        ###transzero
        # nclass=200 #cub
        # nattr = 312 #cub
        # nattr_embed = 512 #cub
        # max_words = 180#cub

        # nclass=50# awa2
        # nattr = 85 #awa2
        # nattr_embed = 512 #awa2
        # max_words = 180#awa2

        nclass=717# sun
        nattr = 102 #sun
        nattr_embed = 512 #sun
        max_words = 180 #sun
        # self.seenclass = torch.Tensor([x for x in range(150)]).cuda()
        # self.unseenclass = np.array([x for x in range(150, 200)])


        self.weight_ce = torch.nn.Parameter(torch.eye(nclass), requires_grad=False)
        self.is_bias = True
        self.is_conservative = True
        self.log_softmax_func = torch.nn.LogSoftmax(dim=1)


        # fashion text embedding
        self.bert_config = BertConfig.from_json_file("../bert/bert_config.json")   
        self.text_encoder = BertModel(config=self.bert_config, add_pooling_layer=False) 
        text_width = self.text_encoder.config.hidden_size
        embed_dim = 8
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.combine_text_proj = nn.Linear(embed_dim, embed_dim)
        # self.text_encoder_m = BertModel(config=self.bert_config, add_pooling_layer=False)     
        self.decoder_layer = BertOnlyMLMHead(config=self.bert_config)
        self.replace_predict_layer = Replace_Predictor(config=self.bert_config)

        # self.crossentropy_func1 = nn.CrossEntropyLoss()
        # self.crossentropy_func2 = nn.CrossEntropyLoss(weight=torch.Tensor([1,10]))
        

        

        
        
        self.attr_text_proj = nn.Linear(max_words, nattr)
        self.attr_embed_proj = nn.Linear(768, nattr_embed)


        self.alpha = torch.nn.Parameter(torch.Tensor([1]), requires_grad=True)

    def compute_reg_loss(self, in_package):
        att = in_package['att']
        
        tgt = torch.matmul(in_package['batch_label'], att)

        embed = in_package['embed']
        loss_reg = F.mse_loss(embed, tgt, reduction='mean')
        return loss_reg
    
    def compute_loss_transzero(self, in_package):
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]


        #loss_cal = 0#self.compute_loss_Self_Calibrate(in_package)
        loss_reg = self.compute_reg_loss(in_package)


        # ################################## fashion loss  ##################################
        # sim_i2t = F.normalize(in_package['im_to_att_embedding'],dim=1)
        # sim_t2i = F.normalize(in_package['text_to_att_embedding'],dim=1)

        # loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_t2i,dim=1).mean()
        # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_i2t,dim=1).mean()
        # loss_ita = (loss_i2t+loss_t2i)/2
        # w_ita = 0#.1#5


        # Trp loss
        # replace_logit = self.replace_predict_layer(replace_embedding)
        # crossentropy_func = nn.CrossEntropyLoss()
        # mask_loss = crossentropy_func(mask_logit.view(-1,self.bert_config.vocab_size), mask_labels.view(-1))
        # replace_loss = crossentropy_func(replace_logit.view(-1, self.args.replace_kind_num), replace_labels.view(-1))


        ### for cub
        # lambda_ = 0.5
        # lambda_reg = 0.005
        # w_ce = 0.1


        ### sun
        #lambda_ = 0.5
        lambda_reg = 0.007#0.005   0.007for awa2 cub

        loss = lambda_reg * loss_reg #w_ita*loss_ita + 
        #lambda_ * loss_cal +  \ #lambda_reg * loss_reg +
            
        return loss#out_package
    
    # def compute_loss_fashion(self,):

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


    def new_contra_loss(self, x1,x2,y,margin=2.0):
        #print(x1.shape,x2.shape,y) torch.Size([128]) torch.Size([128]) 1
        cos_similarity = F.cosine_similarity(x1, x2,dim=0)
        #print(cos_similarity,cos_similarity.shape)
        loss_contrastive = torch.mean((1-y) * torch.pow(cos_similarity, 2) +
                                      (y) * torch.pow(torch.clamp(margin - cos_similarity, min=0.0), 2))
        return loss_contrastive

    def attr_contra_loss(self, attr_embed, attr_labels, cls_labels, pos_th):
        #torch.Size([64, 102, 128]) torch.Size([102, 717]) torch.Size([64]
        bs = attr_embed.shape[0]
        attr_dim = attr_embed.shape[1]

        attr_labels = attr_labels.t()
        data_attr_labels = attr_labels[cls_labels]  #64,102
        #print(data_attr_labels,data_attr_labels.shape)

        # t = data_attr_labels[0]-data_attr_labels[1]
        # print(torch.max(t),torch.min(t),torch.mean(t))  #0.047,-0.046,-0.0024 for sun
        #                                                 #3   -4    0  for cub
        #                                                 # 1.6  -1.8   0         for awa2
        # bb
        #构建所有的pos对
        loss = 0
        count = 0
        count1 = 0
        for i in range(attr_dim):#遍历属性的每一位
            matched = [0 for _ in range(bs)] #记录是否match过
            for j in range(bs-1):
                if matched[j]==1:
                    continue
                for k in range(j+1,bs):
                    if matched[k]==1:
                        continue
                    else:
                        embed1 = attr_embed[j][i]
                        embed2 = attr_embed[k][i]
                        label = 0
                        if torch.abs(data_attr_labels[j][i]-data_attr_labels[k][i])<pos_th:
                            label = 1
                            count1+=1
                        loss_one = self.new_contra_loss(embed1, embed2, label)
                        #print(loss_one)
                        #bb
                        loss+=loss_one
                        count+=1
                        matched[k]=1
                        if count > bs*3:
                            return loss/max(count,1)

        loss = loss/max(count,1)
        return loss

    def compute_loss_fashion_text(self, text_input_ids, attention_mask, mask_labels, replace_labels):

        # fashion text embedding
        text_input_ids = text_input_ids.cuda()
        attention_mask = attention_mask.cuda()
        mask_labels = mask_labels.cuda()
        replace_labels = replace_labels.cuda()
        text_output = self.text_encoder(text_input_ids, attention_mask = attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        # text_feat = F.normalize(self.combine_text_proj(self.text_proj(text_embeds[:,0,:])), dim=-1)
       
        fusion_embeds = text_embeds
        replace_embedding = fusion_embeds
        mask_logit = self.decoder_layer(fusion_embeds)
        replace_logit = self.replace_predict_layer(replace_embedding)
        crossentropy_func = nn.CrossEntropyLoss()
        # tmp = mask_labels
        # tmp[tmp>0]=1
        # tmp[tmp<0]=0
        #print(replace_labels.shape, torch.sum(replace_labels))
        # bbb
        #print(mask_logit.shape, mask_labels.shape) [12, 30, 30522]) torch.Size([12, 30]
        #print(replace_logit,replace_labels, torch.sum(replace_labels))
        #print(replace_logit.shape, replace_labels.shape) [12, 30, 2]) torch.Size([12, 30]
        #bb
        # mask_logit = torch.nn.functional.softmax(mask_logit, dim=-1)
        # replace_logit = torch.nn.functional.softmax(replace_logit, dim=-1)
        # mask_labels[mask_labels<0]=0
        # print(mask_logit,mask_labels)
        # print(replace_logit,replace_labels)
        # bb
        # print(mask_logit, mask_labels)
        # print(mask_logit.shape, mask_labels.shape)
        mask_loss = crossentropy_func(mask_logit.view(-1,self.bert_config.vocab_size), mask_labels.view(-1))
        # print(mask_loss)
        # print("---")
        # print(replace_logit, replace_labels)
        # print(replace_logit.shape, replace_labels.shape)
        replace_loss = crossentropy_func(replace_logit.view(-1, 2), replace_labels.view(-1))
        # print(replace_loss)
        # bbb
        return mask_loss, replace_loss
        # return mask_loss
    

    def compute_loss_semantics_dist(self, img_attr_embeds, text_input_ids, text_attention_mask, labels):
        """
        img_attr_embed:  bs,attr_num,attr_embed_dim  e.g.  12,312,128 for cub

        """
        bs,attr_num,embed_dim = img_attr_embeds.shape
        img_attr_embeds = F.normalize(img_attr_embeds, dim=-1)

        # fashion text embedding
        mask_token_id = 103    
        text_dropout = 0.1
        #print(text_input_ids.shape) [12,180]  bs,lenth
        # bb
        # print(text_input_ids[0])
        for i in range(len(text_input_ids)):
            for j in range(len(text_input_ids[i])):
                if text_input_ids[i][j]!=mask_token_id:
                    if (random.random() < text_dropout):
                        text_input_ids[i][j] = mask_token_id
        # print(text_input_ids[0])
        # bbb
        text_input_ids = text_input_ids.cuda()
        text_attention_mask = text_attention_mask.cuda()

        text_output = self.text_encoder(text_input_ids, attention_mask = text_attention_mask,                      
                                        return_dict = True, mode = 'text')
                
        text_embeds = text_output.last_hidden_state
        #print(text_embeds.shape) #[bs, 180, 768]  bs,句长，word embedding维度
        
        text_attr_embeds = self.attr_embed_proj(text_embeds)
        text_attr_embeds = text_attr_embeds.permute(0,2,1)
        text_attr_embeds = self.attr_text_proj(text_attr_embeds)
        text_attr_embeds = text_attr_embeds.permute(0,2,1)
        #print(text_attr_embeds.shape)
        text_attr_embeds = F.normalize(text_attr_embeds, dim=-1)
        #bb
        # text_feat = F.normalize(self.combine_text_proj(self.text_proj(text_embeds[:,0,:])), dim=-1)
        #print(img_attr_embeds.shape, text_attr_embeds.shape)
        #[12, 102, 512]) torch.Size([12, 102, 512]
        
        #1. 直接拉近img attr和text attr的距离
        #mse
        # img_attr_embeds = img_attr_embeds.reshape(img_attr_embeds.shape[0],-1)
        # text_attr_embeds = text_attr_embeds.reshape(text_attr_embeds.shape[0],-1)
        # loss = torch.mean((img_attr_embeds - text_attr_embeds)**2)
        #cos
        # loss = F.cosine_similarity(img_attr_embed, text_attr_embeds, dim=-1)
        # loss = torch.mean(loss)
        #transzero
        # loss_i2t = -torch.sum(F.log_softmax(img_attr_embeds, dim=-1)*text_attr_embeds,dim=-1).mean()
        # loss_t2i = -torch.sum(F.log_softmax(text_attr_embeds, dim=-1)*img_attr_embeds,dim=-1).mean()
        # loss = torch.abs((loss_i2t+loss_t2i)/2)
    
        #2. 对齐不同数据的 img attr距离和  text_attr的距离
        # print(labels)
    
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
                    # print(loss_one, self.alpha)
                    # bb
                #loss_one*=self.alpha[0] #缩放一定的倍数
                loss += loss_one
                count+=1
        # bb
        loss/=count  
        # print(loss,count)
        # bb
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


        #print(logits.shape, code_logits.shape, labels.shape, attr_data.shape, img_attr_embed.shape)
        #[12, 717] [12, 64] [12] [102, 717] [12, 102, 512]
        
      
        ### transzero 
        # transzero_loss = self.compute_loss_transzero(package)
        # att = package['att']
        # #print(att.shape,package['batch_label'].shape)   [717, 102]) torch.Size([12]
        
        # tgt = torch.matmul(package['batch_label'], att)
        # embed = package['embed']
        # loss_reg = F.mse_loss(embed, tgt, reduction='mean')

        # transzero_w = 0.5*0.007 #0.5 for cub awa2,  0.3for sun
        # transzero_loss = loss_reg*transzero_w


        embed = package['my_img_embed']
        att = package['att']
        if len(labels.size()) == 1:
            labels_onehot = self.weight_ce[labels]
        # tgt = torch.matmul(package['batch_label'], att)
        tgt = torch.matmul(labels_onehot, att)
        loss_reg = F.mse_loss(embed, tgt, reduction='mean')

        transzero_w = 2#*0.007 #0.5 for cub awa2,  0.3for sun
        transzero_loss = loss_reg*transzero_w




       
        margin_logits = self.get_margin_logits(logits, labels)
        ce = F.cross_entropy(margin_logits, labels)

        # hd = self.get_hamming_distance()
        # triu = torch.ones_like(hd).bool()
        # triu = torch.triu(triu, 1)

        semantics_dist_loss = self.compute_loss_semantics_dist(img_attr_embed, text_input_ids, text_attention_mask, labels)
        w_se = 0# 3 for awa2  4for cub
        semantics_dist_loss*=w_se
        # text_mask_loss, text_replace_loss = self.compute_loss_fashion_text(text_input_ids, text_attention_mask, text_mask_labels, text_replace_labels)
        # w_tm = 1
        # w_tr = 1
        # # # print(text_mask_loss)
        # text_mask_loss *= w_tm
        # text_replace_loss *= w_tr
        # print('\n',ce,transzero_loss, semantics_dist_loss)
        # bb
        self.losses['ce'] = ce#*0.1

        loss = ce +  semantics_dist_loss+transzero_loss #+ \
                
                # text_mask_loss + text_replace_loss

        return loss
