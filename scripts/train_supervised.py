import torch

from functions.hashing import get_hamm_dist
import time
# from torchstat import stat

def get_output_and_loss_supervised(model, criterion, data, labels, index, onehot, 
                    loss_name, loss_cfg, stage, no_loss, attr_data, label_a, label_v, middle_graph,
                    text_input_ids, text_attention_mask, text_mask_labels, text_replace_labels,
                    B,S,omega):
    #logits, code_logits = model(data)[:2]
    t = time.time()
    logits, code_logits, new1, package = model(data, attr_data)

    # _input_size = (3,224,224)
    # stat(model,_input_size)

    # print("infer time:", time.time()-t)
    # print(labels.shape, label_a.shape, label_v.shape)
    # bb
    # package['batch_label'] = label_v
    # package['vec_bias'] = model.vec_bias
    package['att'] = model.att
    if no_loss:
        loss = torch.tensor(0.)
    else:
        loss = criterion(logits, code_logits, labels, 
                    attr_data, new1,
                    text_input_ids, text_attention_mask, text_mask_labels, text_replace_labels,
                    package,
                    onehot=onehot)
    return {
               'logits': logits,
               'code_logits': code_logits
           }, loss

##origin  jmlh
# def get_output_and_loss_supervised(model, criterion, data, labels, index, onehot, 
#                     loss_name, loss_cfg, stage, no_loss, attr_data, label_a, label_v, middle_graph,
#                     text_input_ids, text_attention_mask, text_mask_labels, text_replace_labels,
#                     B,S,omega):
#     logits, code_logits = model(data)[:2]
#     # logits = model(data)

#     if no_loss:
#         loss = torch.tensor(0.)
#     else:
#         loss = criterion(logits, code_logits, labels,
#                     onehot=onehot)
#     return {
#                'logits': logits,
#                'code_logits': code_logits
#            }, loss

def update_meters_supervised(model, meters, out, labels, onehot, criterion, loss_name, loss_cfg):
    logits = out['logits']
    code_logits = out['code_logits']
    acc = calculate_accuracy(logits, labels, loss_cfg.get('multiclass', False), onehot=onehot)

    for key in criterion.losses:
        val = criterion.losses[key]
        if hasattr(val, 'item'):
            val = val.item()
        meters[key].update(val)

    if loss_name in ['csq', 'dpn', 'orthocos', 'orthoarc']:
        hd = get_hamm_dist(code_logits, model.centroids)
        hdacc = calculate_accuracy(-hd, labels, loss_cfg.get('multiclass', False), onehot=onehot)
        meters['hdacc'].update(hdacc.item())

    meters['acc'].update(acc.item())


def calculate_accuracy(logits, labels, mmclass, onehot=True):
    if mmclass:
        acc = torch.tensor(0.)
        # pred = logits.topk(5, 1, True, True)[1].t()
        # correct = pred.eq(labels.argmax(1).view(1, -1).expand_as(pred))
        # acc = correct[:5].view(-1).float().sum(0, keepdim=True) / logits.size(0)
    else:
        if onehot:
            acc = (logits.argmax(1) == labels.argmax(1)).float().mean()
        else:
            acc = (logits.argmax(1) == labels).float().mean()  # logits still is one hot encoding
    #print(logits.shape, labels.shape) #torch.Size([64, 200]) torch.Size([64, 200]
    # print(acc)
    # bb
    return acc