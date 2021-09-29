#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:00:21 2018

@author: ws
"""
#此代码提供一些工具函数

import torch
import numpy as np
import os
import json
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from ipdb import set_trace

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
    

#此函数用于把预测的predictions  由索引表示转换为word表示
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out


#不使用强化时候计算损失函数
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
#        alpha = 0.25
#        beta = 2
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
#        pt = torch.softmax(input,dim=2)
#        output_pt = pt.gather(2, target.unsqueeze(2))
#        output_pt = output_pt.squeeze(2)
#        weights = alpha*(1-output_pt)**beta
        #output = torch.sum(output_pt)/torch.sum(mask)
        #set_trace()
        output = -input.gather(2, target.unsqueeze(2).long())
        output = output.squeeze(2) 
#        output = output*weights* mask
        output = output* mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


#使用奖励（强化学习）时候的损失函数        
class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):            #input:预测的的log概率  seq:生成的caption
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output




#transformer专用优化器
class NoamOpt(object):
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


        
def get_std_opt(model, factor=1, warmup=2000):
    return NoamOpt(model.module.model.tgt_embed[0].d_model, factor, warmup,
            torch.optim.Adam(model.module.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
        
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x



def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def var_wrapper(x, cuda=True, volatile=False):
    if type(x) is dict:
        return {k: var_wrapper(v, cuda, volatile) for k,v in x.items()}
    if type(x) is list or type(x) is tuple:
        return [var_wrapper(v, cuda, volatile) for v in x]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if cuda:
        x = x.cuda()
    else:
        x = x.cpu()
    if torch.is_tensor(x):
        x = Variable(x, volatile=volatile)
    if isinstance(x, Variable) and volatile!=x.volatile:
        x = Variable(x.data, volatile=volatile)
    return x
