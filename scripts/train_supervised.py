import torch

from functions.hashing import get_hamm_dist
import time
# from torchstat import stat

def get_output_and_loss_supervised(model, criterion, data, labels, index, onehot, 
                    loss_name, loss_cfg, stage, no_loss, attr_data, label_a, label_v, middle_graph,
                    text_input_ids, text_attention_mask, text_mask_labels, text_replace_labels,
                    B,S,omega):

    t = time.time()
    logits, code_logits, new1, package = model(data, attr_data)

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
    else:
        if onehot:
            acc = (logits.argmax(1) == labels.argmax(1)).float().mean()
        else:
            acc = (logits.argmax(1) == labels).float().mean()  # logits still is one hot encoding

    return acc