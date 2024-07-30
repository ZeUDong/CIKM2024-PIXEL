# Angular deep supervised hashing for image retrieval
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
from functions.loss.base_cls import BaseClassificationLoss


# cvpr2021 clear those instances that have no positive instances to avoid training error
class SupConLoss_clear(torch.nn.Module):
    def __init__(self, temperature=10.0):
        super(SupConLoss_clear, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos*(1-single_samples)
        loss = loss.sum()/(loss.shape[0]-single_samples.sum())

        return loss


# ijcai21
class hash(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data

        return grad_output

def hash_layer(input):
    return hash.apply(input)

# ijcai21
class NtXentLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature):
        super(NtXentLoss, self).__init__()
        #self.batch_size = batch_size
        self.temperature = temperature
        #self.device = device

        #self.mask = self.mask_correlated_samples(batch_size)
        self.similarityF = torch.nn.CosineSimilarity(dim = 2)
        self.criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')
    

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    

    def forward(self, z_i, z_j, device):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        #sim = 0.5 * (z_i.shape[1] - torch.tensordot(z.unsqueeze(1), z.T.unsqueeze(0), dims = 2)) / z_i.shape[1] / self.temperature

        sim_i_j = torch.diag(sim, batch_size )
        sim_j_i = torch.diag(sim, -batch_size )
        
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)

        labels = torch.zeros(N).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

#ECCV2022 SEMICON 
class ADSH_Loss(torch.nn.Module):
    
    def __init__(self, code_length, gamma=200):
        super(ADSH_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma

    def forward(self, F, B, S, omega):
        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum() / (F.shape[0] * B.shape[0]) / self.code_length * 12
        quantization_loss = ((F - B[omega, :]) ** 2).sum() / (F.shape[0] * B.shape[0]) * self.gamma / self.code_length * 12

        loss = hash_loss + quantization_loss
        return loss, hash_loss, quantization_loss







class ADSHLoss(BaseClassificationLoss):
    def __init__(self, alpha=1, beta=1, s=10, m=0.2, multiclass=False, method='cosface', **kwargs):
        super().__init__()
        # print(kwargs)
        # bbb
        logging.info("Loss parameters are: ", locals())
        self.alpha = alpha
        self.beta = beta
        self.s = s
        self.m = m

        self.method = method
        self.multiclass = multiclass
        self.weight = None

        # self.nclass = nclass

        #nips2020
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_regre = torch.nn.MSELoss()

        #cvpr2021
        self.contras_criterion = SupConLoss_clear()

        #ijcai21
        batch_size = 64
        temperature = 0.3
        self.criterion_ijcai21 = NtXentLoss(batch_size, temperature)


        self.criterion_eccv2022_semicon = ADSH_Loss(code_length=64)

        ###transzero
        #print(kwargs)
        nclass=200 #cub
        self.seenclass = torch.Tensor([x for x in range(150)]).cuda()
        self.unseenclass = np.array([x for x in range(150,200)])

        # nclass=717 #sun
        # self.seenclass = torch.Tensor([x for x in range(500)]).cuda()
        # self.unseenclass = np.array([x for x in range(500,717)])

        self.weight_ce = torch.nn.Parameter(torch.eye(nclass), requires_grad=False)
        self.is_bias = True
        self.is_conservative = True
        self.log_softmax_func = torch.nn.LogSoftmax(dim=1)


    ### transzero loss
    def compute_loss_Self_Calibrate(self, in_package):
        S_pp = in_package['pred']
        Prob_all = F.softmax(S_pp, dim=-1)
        #print(self.unseenclass)
        Prob_unseen = Prob_all[:, self.unseenclass]
        assert Prob_unseen.size(1) == len(self.unseenclass)
        mass_unseen = torch.sum(Prob_unseen, dim=1)
        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp

    def compute_aug_cross_entropy(self, in_package):
        Labels = in_package['batch_label']
        S_pp = in_package['pred']
        
        vec_bias = in_package['vec_bias']

        if self.is_bias:
            S_pp = S_pp - vec_bias

        if not self.is_conservative:
            S_pp = S_pp[:, self.seenclass]
            Labels = Labels[:, self.seenclass]
            assert S_pp.size(1) == len(self.seenclass)

        Prob = self.log_softmax_func(S_pp)

        loss = -torch.einsum('bk,bk->b', Prob, Labels)
        loss = torch.mean(loss)
        return loss

    def compute_reg_loss(self, in_package):
        att = in_package['att']
        tgt = torch.matmul(in_package['batch_label'], att)
        embed = in_package['embed']
        loss_reg = F.mse_loss(embed, tgt, reduction='mean')
        return loss_reg

    def compute_loss_transzero(self, in_package):
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]

        # print(in_package['pred'].shape, in_package['batch_label'].shape)
        # import time
        # time.sleep(10000)

        loss_CE = 0#self.compute_aug_cross_entropy(in_package)
        loss_cal = 0#self.compute_loss_Self_Calibrate(in_package)
        loss_reg = self.compute_reg_loss(in_package)
        # bb
        sim_i2t = F.normalize(in_package['im_to_att_embedding'],dim=1)
        sim_t2i = F.normalize(in_package['text_to_att_embedding'],dim=1)

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_t2i,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_i2t,dim=1).mean()
        #print(loss_i2t,loss_t2i)
        loss_ita = (loss_i2t+loss_t2i)/2
        w_ita = 0.1
        #bb

        ### for cub
        # lambda_ = 0.5
        # lambda_reg = 0.005
        # w_ce = 0.1

        ### sun
        lambda_ = 0.5
        lambda_reg = 0.005
        w_ce = 0.1

        loss = w_ce*loss_CE + lambda_ * \
            loss_cal + lambda_reg * loss_reg + \
            w_ita*loss_ita
        # out_package = {'loss': loss, 'loss_CE': loss_CE,
        #                'loss_cal': loss_cal, 'loss_reg': loss_reg}
        return loss#out_package


    def get_hamming_distance(self):
        assert self.weight is not None, 'please set weights before calling this function'

        b = torch.div(self.weight, self.weight.norm(p=2, dim=-1, keepdim=True) + 1e-7)
        b = torch.tanh(b)

        nbit = b.size(1)

        hd = 0.5 * (nbit - b @ b.t())
        return hd

    # def get_margin_logits(self, logits, labels):
    #     y_onehot = torch.ones_like(logits)
    #     y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
    #
    #     norm = logits.norm(p=2, dim=-1, keepdim=True)
    #     theta = torch.div(logits, norm + 1e-7).acos().clamp(-0.999999, 0.999999)
    #
    #     theta += ()
    #
    #     margin_logits = y_onehot * logits
    #
    #     return margin_logits

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

    def nips2020attrloss(self,pre_attri, pre_class,label_a,label_v,attention,middle_graph):
        #xe 1 --attri 1e-4   regular 0.0005
        #l_xe 0 --l_attri 1e-2 --l_regular 0.5e-6
        #cpt 2e-9
        xe = 1
        attri = 1e-4
        cpt = 2e-9
        loss = 0
        if xe > 0:
            loss_xe = xe * self.criterion(pre_class, label_v)
            loss = loss_xe

        if attri > 0:
            loss_attri = attri * self.criterion_regre(pre_attri, label_a)
            loss += loss_attri

        if cpt > 0:
            batch_size, attri_dim, map_dim, map_dim = attention.size()
            peak_id = torch.argmax(attention.view(batch_size * attri_dim, -1), dim=1)
            peak_mask = middle_graph[peak_id, :, :].view(batch_size, attri_dim, map_dim, map_dim)
            cpt_loss = cpt * torch.sum(torch.sigmoid(attention) * peak_mask)
            loss += cpt_loss

        return loss

    def compute_kl(self, prob, prob_v):
        #for ijcai21loss
        prob_v = prob_v.detach()
        # prob = prob.detach()

        kl = prob * (torch.log(prob + 1e-8) - torch.log(prob_v + 1e-8)) + (1 - prob) * (torch.log(1 - prob + 1e-8 ) - torch.log(1 - prob_v + 1e-8))
        kl = torch.mean(torch.sum(kl, axis = 1))
        return kl

    def ijcai21loss(self, code_logits, label_v):
        #print(code_logits.shape, label_v.shape)[64, 64]) torch.Size([64]
        #print(label_v)
        #获取次数大于1的类别
        unique_tensor, counts = torch.unique(label_v, return_counts=True)
        repeated_indices = torch.where(counts > 1)[0]
        repeated_elements = unique_tensor[repeated_indices]
        #print(repeated_elements)
        loss_all = 0
        for cid in range(len(repeated_elements)):
            same_codes = code_logits[label_v==repeated_elements[cid]]
            
            prob_i = torch.sigmoid(same_codes[0]).unsqueeze(0)
            z_i = hash_layer(prob_i - 0.5)

            for jid in range(1,len(same_codes)):
                prob_j = torch.sigmoid(same_codes[jid]).unsqueeze(0)
                z_j = hash_layer(prob_j - 0.5)

                kl_loss = (self.compute_kl(prob_i, prob_j) + self.compute_kl(prob_j, prob_i)) / 2
                contra_loss = self.criterion_ijcai21(z_i, z_j, code_logits.device)
                #print(self.hparams.weight) 0.001
                weight = 0.001
                loss = contra_loss + weight * kl_loss
                loss_all += loss

        return loss_all


    def class_scores_for_loop(self, embed, input_label,attribute_seen, relation_net):
        #cvpr2021 cls
        cls_temp = 1.0
        nclass_seen = self.nclass#这里seen应该是40，但是oneloss这个代码没有区分，这里还是直接用总的50，只是只有40类的数据，一样的
        all_scores=torch.FloatTensor(embed.shape[0],nclass_seen).cuda()
        for i, i_embed in enumerate(embed):
            #print(i_embed.shape)
            expand_embed = i_embed.repeat(nclass_seen, 1)#.reshape(embed.shape[0] * opt.nclass_seen, -1)
            #print(expand_embed.shape, attribute_seen.shape)
            all_scores[i]=(torch.div(relation_net(torch.cat((expand_embed, attribute_seen.t().float()), dim=1)),cls_temp).squeeze())
        score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
        # normalize the scores for stable training
        scores_norm = all_scores - score_max.detach()
        mask = F.one_hot(input_label, num_classes=nclass_seen).float().cuda()
        exp_scores = torch.exp(scores_norm)
        log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
        cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
        return cls_loss


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
        #print(attr_labels.shape)
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



    def forward(self, logits, code_logits, labels, 
        attr_data, new1,
        package=None,
        onehot=True):
        if self.multiclass:
            logits, labels = self.get_dynamic_logits(logits, code_logits, labels)
            labels = labels.argmax(1)
        else:
            if onehot:
                labels = labels.argmax(1)

        #print(label_v.shape,logits.shape)
        #label_v:每条数据对应的class id  非onehot
        # print("222 ",package['batch_label'].shape)
        ### transzero 
        transzero_loss = self.compute_loss_transzero(package)
        transzero_w = 0.1
        transzero_loss *= transzero_w

        ### nips2020
        # attr_w = 0#2for cub  0.5for awa2  2for sun
        # self.attrloss = 0#self.nips2020attrloss(pre_attri, pre_class,label_a,label_v,attention,middle_graph)

        # ### cvpr2021
        # ins_w = 0#2#1for cub   0.2for awa2   4for sun
        # # cls_w = 0#1for cub   0.2for awa2   4for sun
        # real_ins_contras_loss = self.contras_criterion(outz_real, label_v)
        # if torch.isnan(real_ins_contras_loss):
        #     real_ins_contras_loss = 0
        # cls_loss_real = self.class_scores_for_loop(embed_real, label_v, attr_data, Dis_Embed_Att)


        # ### IJCAI21
        # icjai_w = 0#0.1for cub   0.1for awa2   6for sun
        # icjai21loss = self.ijcai21loss(code_logits, label_v)


        # ### ECCV2022
        # adsh_w = 0 #0.3#1for cub    1for awa2   for sun
        # adsh_loss = 0 
        # if B is not None:
        #     adsh_loss, _, _ = self.criterion_eccv2022_semicon(code_logits, B, S, omega)

        #attr contra
        # new1_loss = self.attr_contra_loss(new1, attr_data, labels, pos_th = 2)
        # new1_w = 1
        # print(torch.max(t),torch.min(t),torch.mean(t))  #0.047,-0.046,-0.0024  0.03 for sun
        #                                                 #3   -4    0        2  for cub
        #                                                 # 1.6  -1.8   0     0.5 for awa2

        margin_logits = self.get_margin_logits(logits, labels)
        ce = F.cross_entropy(margin_logits, labels)

        # hd = self.get_hamming_distance()
        # triu = torch.ones_like(hd).bool()
        # triu = torch.triu(triu, 1)

        # hd_triu = hd[triu]

        # meanhd = torch.mean(-hd_triu)
        # varhd = torch.var(hd_triu)

        self.losses['ce'] = ce
        # self.losses['meanhd'] = meanhd
        # self.losses['varhd'] = varhd

        loss = ce +  \
                transzero_loss
                #adsh_w*adsh_loss

        return loss
