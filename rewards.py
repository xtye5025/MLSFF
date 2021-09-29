#coding:utf-8

import sys
sys.path.append("../../../../metric/coco_caption_3/pycocoevalcap/ciderD")
sys.path.append("../../../../metric/coco_caption_3/pycocoevalcap/bleu")

import numpy as np
from collections import OrderedDict
import torch
from ciderD import CiderD
#from bleu import Bleu

CiderD_scorer = None
#Bleu_scorer = None


#利用预处理的缓存初始化CIDER计算器
def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
#    global Bleu_scorer
#    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

#用于计算奖励
def get_self_critical_reward(model, fc_feats, att_feats,attr_feats, att_masks, data, gen_result, opt):
#    batch_size = gen_result.size(0)
    batch_size = gen_result.size(0)
    seq_per_img = batch_size // len(data['gts'])                               #每一幅图像的caption数目
    

    model.eval()
    with torch.no_grad():
        greedy_res, _ = model.module.sample(fc_feats, att_feats, attr_feats, att_masks,opt)   #按照贪婪算法得到的结果
    model.train()

    res = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]                    #把按照贪婪算法和按照概率取的结果拼了起来

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}#重复了两遍batch_size的内容
    if opt['cider_reward_weight'] > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)               #计算按照当前的gt 和当前贪婪算法和按照概率取的结果计算cider 
#        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt['bleu_reward_weight'] > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt['cider_reward_weight'] * cider_scores + opt['bleu_reward_weight'] * bleu_scores  #加权 得到的sores是两个batch_size的结果

    scores = scores[:batch_size] - scores[batch_size:]                         #贪婪算法的结果作为baseline 计算sores

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)         #repeat 从(1280,)到（1280,16） 得到最后的奖励

    return rewards    




