#coding:utf-8

import torch
import numpy as np
import json
import os
from ipdb import set_trace
import tools.utils as utils
import sys
from ipdb import set_trace
sys.path.append('../../../../metric/')
from settings import for_eval_json
from coco_caption_3.pycocotools.coco import COCO
from coco_caption_3.pycocoevalcap.eval import COCOEvalCap

def language_eval(args,preds, model_id, split):
    annFile =for_eval_json 


    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', args.pretrained_model_path + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    #有多少张图片被预测
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) 

    #计算指标
    cocoRes = coco.loadRes(preds_filt)                                         #pre_result
    cocoEval = COCOEvalCap(coco, cocoRes)                                      #pre and gt    to eval
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    
    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
        
        
    #存储可视化结果    
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(args,model, crit, loader, eval_kwargs):
    num_images = eval_kwargs['val_images_use']
    split = eval_kwargs['split']



    model.eval()
    loader.reset_iterator(split)   #重置迭代器

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    tem=0
    while True:
        data = loader.get_batch(split)
        
        data['fc_feats']= data['att_feats'].mean(1)                            #取平均值作为fc feature
        
        n = n + loader.batch_size


        #计算在验证集上的损失
        if data.get('labels', None) is not None :
            tmp = [data['fc_feats'], data['att_feats'], data['attr_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, attr_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
#                loss = crit(model(fc_feats, att_feats, labels, att_masks)[0], labels[:,1:], masks[:,1:]).item()
                 #logits=model(fc_feats,att_feats,labels,att_masks)                                
                 #loss= crit(logits, labels[:,1:],masks[:,1:].float()).item()
                 logits=model(fc_feats,att_feats,attr_feats,labels,att_masks)                                
                 loss= crit(logits,labels[:,1:],masks[:,1:].float()).item()  
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1



        #生成预测的结果predictions
        data['att_masks']=None
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['attr_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, attr_feats, att_masks = tmp
        with torch.no_grad():
            seq = model.module.sample(fc_feats, att_feats, attr_feats, att_masks, eval_kwargs)[0].data
        #set_trace()
        sents = utils.decode_sequence(loader.ix_to_word, seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 1) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            #print(entry)
            predictions.append(entry)




        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()                                                  #超过验证集图像数目的话，多出来的不要
            
        if ix0-tem>500:
            tem=ix0
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))
        #所有的都已经预测完
        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break


    lang_stats = language_eval(args,predictions, eval_kwargs['net'], split)
    



    model.train()
    return loss_sum/loss_evals, predictions, lang_stats
