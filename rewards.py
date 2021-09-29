#coding:utf-8

import sys
sys.path.append("../../../../metric/coco_caption_3/pycocoevalcap/ciderD")
sys.path.append("../../../../metric/coco_caption_3/pycocoevalcap/bleu")

import numpy as np
from collections import OrderedDict
import torch
from ciderD import CiderD
CiderD_scorer = None

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(model, fc_feats, att_feats,attr_feats, att_masks, data, gen_result, opt):
    batch_size = gen_result.size(0)
    seq_per_img = batch_size // len(data['gts'])                           
    

    model.eval()
    with torch.no_grad():
        greedy_res, _ = model.module.sample(fc_feats, att_feats, attr_feats, att_masks,opt)
    model.train()

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]                    

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if opt['cider_reward_weight'] > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)              
    else:
        cider_scores = 0
    if opt['bleu_reward_weight'] > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt['cider_reward_weight'] * cider_scores + opt['bleu_reward_weight'] * bleu_scores  

    scores = scores[:batch_size] - scores[batch_size:]                         

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)       

    return rewards    




