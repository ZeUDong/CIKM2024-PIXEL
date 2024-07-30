import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import register_network, BaseArch
from models.architectures.helper import get_hash_fc_with_normalizations, get_backbone
from models.layers.cossim import CosSim
from transformers.activations import ACT2FN


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", SAtt=True):
        super(TransformerDecoderLayer, self).__init__(d_model, nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout,
                                                      activation=activation)
        self.SAtt = SAtt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt : attribute embedding 
        # memory: image embedding
        if self.SAtt:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class Transformer(nn.Module):
    def __init__(self, ec_layer=1, dc_layer=1, dim_com=300,
                 dim_feedforward=2048, dropout=0.1, heads=1,
                 in_dim_cv=2048, in_dim_attr=300, SAtt=True,
                 aux_embed=True):
        super(Transformer, self).__init__()
        # input embedding
        self.embed_cv = nn.Sequential(nn.Linear(in_dim_cv, dim_com))
        if aux_embed:
            self.embed_cv_aux = nn.Sequential(nn.Linear(in_dim_cv, dim_com))
        self.embed_attr = nn.Sequential(nn.Linear(in_dim_attr, dim_com))
    
        # transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model=dim_com,
                                                nhead=heads,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                SAtt=SAtt)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=dc_layer)


    def forward(self, f_cv, f_attr):
        # print(f_cv.shape, f_attr.shape)
        #([12, 2048, 49]) torch.Size([312, 300])
        # linearly map to common dim
        h_cv = self.embed_cv(f_cv.permute(0, 2, 1))
        h_attr = self.embed_attr(f_attr)
        h_attr_batch = h_attr.unsqueeze(0).repeat(f_cv.shape[0], 1, 1)
        # visual encoder
        # attribute-visual decoder
        out = self.transformer_decoder(h_attr_batch.permute(1, 0, 2), h_cv.permute(1, 0, 2))
        return out.permute(1, 0, 2), h_cv, h_attr_batch
    




@register_network('orthohash')
class ArchOrthoHash(BaseArch):
    """Arch OrthoHash"""
    def __init__(self, config,  n_attr=0, **kwargs):
        super(ArchOrthoHash, self).__init__(config, **kwargs)
 
        self.dataset = config['dataset']
        ##tranzero
        transzero_att = config['transzero_att']
        transzero_w2v_att = config['transzero_w2v_att'] 
        mask_bias = config['mask_bias'] 
        
        bias=1
        self.bias = nn.Parameter(torch.tensor(bias), requires_grad=False)
        # nclass=200
        # mask_bias = np.ones((1, nclass))
        # mask_bias[:, self.seenclass.cpu().numpy()] *= -1
        self.mask_bias = nn.Parameter(torch.tensor(
            mask_bias, dtype=torch.float), requires_grad=False)
        # class-level semantic vectors
        self.att = nn.Parameter(F.normalize(torch.from_numpy(transzero_att).float().cuda()), requires_grad=False)
        # GloVe features for attributes name
        self.V = nn.Parameter(F.normalize(torch.from_numpy(transzero_w2v_att).float().cuda()), requires_grad=True)
        #300x312 
        self.transformer = Transformer(
            ec_layer=1,
            dc_layer=1,
            dim_com=300,
            dim_feedforward=512,
            dropout=0.4,
            SAtt=True,
            heads=1,
            aux_embed=True)
        # mapping
        dim_v = 300
        self.W_1 = nn.Parameter(nn.init.normal_(
            torch.empty(dim_v, 300)), requires_grad=True)

        hash_layer = config['loss_param'].get('hash_layer', 'identity')
        hash_kwargs = config['loss_param']
        cossim_ce = config['loss_param'].get('cossim_ce', True)
        learn_cent = config['loss_param'].get('learn_cent', True)

        self.backbone = get_backbone(backbone=self.backbone_name,
                                     nbit=self.nbit,
                                     nclass=self.nclass,
                                     pretrained=self.pretrained,
                                     freeze_weight=self.freeze_weight, **kwargs)

        if cossim_ce:
            self.ce_fc = CosSim(self.nbit, self.nclass, learn_cent)
        else:
            self.ce_fc = nn.Linear(self.nbit, self.nclass)

        self.hash_fc = get_hash_fc_with_normalizations(in_features=self.backbone.in_features,
                                                       nbit=self.nbit,
                                                       bias=self.bias,
                                                       kwargs=hash_kwargs)
    
        #nips2020
        self.softmax = nn.Softmax(dim=1)
        self.attr_conv = nn.Conv2d(2048, n_attr, 1, 1) #2048 for resnet101

        ### new add
        self.attr_conv2 = nn.Conv2d(2048, n_attr, 1, 1)
        if self.dataset=='cub':
            self.attr_emb = 512#cub
        elif self.dataset=='awa2':
            self.attr_emb = 512#awa2
        elif self.dataset=='sun':
            self.attr_emb = 512#sun
        self.attr_fc = nn.Linear(49, self.attr_emb)


        ###fashionSAP
        embed_dim = 256
        if self.dataset=='cub':
            attr_num = 312
        elif self.dataset=='sun':
            attr_num = 102
        elif self.dataset=='awa2':
            attr_num = 85
        self.img2embed = nn.Linear(300*49, embed_dim)
        self.text2embed = nn.Linear(attr_num*300, embed_dim)
        
        
        # # ##our
        # img_attr_in = None
        # if self.dataset=='cub':
        #     img_attr_in = 312
        # elif self.dataset=='sun':
        #     img_attr_in = 102
        # elif self.dataset=='awa2':
        #     img_attr_in = 85
        # self.img_attr_hash_fc = nn.Sequential(
        #                                 nn.Linear(img_attr_in, self.nbit, bias=self.bias),
        #                                 # nn.BatchNorm1d(self.nbit)
        #                             )
        # self.img_attr_ce_fc = nn.Linear(self.nbit, self.nclass)





    def get_features_params(self):
        return self.backbone.get_features_params()

    def get_hash_params(self):
        return list(self.ce_fc.parameters()) + list(self.hash_fc.parameters())

    def forward_feature_transformer(self, Fs):
        # visual 
        if len(Fs.shape) == 4:
            shape = Fs.shape
            Fs = Fs.reshape(shape[0], shape[1], shape[2] * shape[3])
        Fs = F.normalize(Fs, dim=1)

        ### fashionSAP
        # attributes
        # locality-augmented visual features
        # fs 是图像 embedding , self.V 是属性embedding
        Trans_out,h_cv,h_attr_batch = self.transformer(Fs, self.V)

        # print(h_cv.shape,h_attr_batch.shape)
        h_cv = h_cv.reshape(h_cv.shape[0],-1)
        h_attr_batch = h_attr_batch.reshape(h_attr_batch.shape[0],-1)
        # print(h_cv.shape,h_attr_batch.shape)
        im_to_att_embedding = self.img2embed(h_cv)
        text_to_att_embedding = self.text2embed(h_attr_batch)
        # print(im_to_att_embedding.shape, text_to_att_embedding.shape)
        # bb
        # embedding to semantic space
        embed = torch.einsum('iv,vf,bif->bi', self.V, self.W_1, Trans_out)
        return embed,im_to_att_embedding,text_to_att_embedding

    # def forward_attribute(self, embed):
    #     embed = torch.einsum('ki,bi->bk', self.att, embed)
    #     self.vec_bias = self.mask_bias*self.bias
    #     embed = embed + self.vec_bias
    #     return embed

    def forward(self, x, attribute):
        #print(x.shape)  [12, 3, 224, 224]
        x,features = self.backbone(x)
        #print(x.shape, features.shape) #[12, 2048]  [12, 2048, 7, 7] 
        #这里是224输入，tranzero是448输入所以是14x1
        #bb
        ### transzero
        # v2s_embed, im_to_att_embedding, text_to_att_embedding = self.forward_feature_transformer(features) 
        # # classification
        # #print("v2s_embed:",v2s_embed.shape) [12, 102]  102=sun attr 
        # #print(im_to_att_embedding.shape, text_to_att_embedding.shape) [12, 256]) [12, 256]
        # package = {'pred': self.forward_attribute(v2s_embed), #VSEN?
        #            'embed': v2s_embed,
        #            'im_to_att_embedding':im_to_att_embedding,
        #            'text_to_att_embedding':text_to_att_embedding}
        # # print(package['pred'].shape) #
        # package['S_pp'] = package['pred']
         
        package = {}
        ## new attr contra
        new1 = self.attr_conv(features) #bs,102,7,7
        new1 = new1.view(new1.shape[0], new1.shape[1], -1) #bs,102,49
        attr_embed = self.attr_fc(new1)  #bs,attr_num,512
        #print("attr_embed.shape: ", attr_embed.shape) [12, 102, 512]
        # bb
        package['my_img_embed'] = attr_embed.mean(2)

        hash_code = self.hash_fc(x)
        class_pre = self.ce_fc(hash_code)

        # hash_code = self.img_attr_hash_fc(package['my_img_embed'])
        # class_pre = self.img_attr_ce_fc(hash_code)
        #print(v.shape,u.shape)
        # torch.Size([bs, 64]) torch.Size([bs, 717]  64=nbit 717=sun
        return class_pre, hash_code, attr_embed,package
