#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torchvision import transforms
import os
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import tools.utils as utils
import time
from torch.autograd import Variable
from dataloader import  DataLoader
import eval_utils
from rewards import init_scorer,get_self_critical_reward
from ipdb import set_trace
import json
from nets.LSFF import LSFF
from RAdam.radam import RAdam
from RAdam.radam import AdamW
import opts                                  
import fitlog
from settings import input_json, cached_tokens, vocab_dir
from tools.init import set_seed, get_device

args=opts.get_args()
fitlog.set_log_dir("logs/")
fitlog.add_hyper(args.drop_prob_lm,"dropout")
fitlog.add_hyper(args.sg_label_embed_size,"attr_size") 
fitlog.add_hyper(args.use_num_attr,"use_attr")
fitlog.add_hyper(args.input_encoding_size,"input_embed")
fitlog.add_hyper(args.scheduled_sampling_start,"ss_start")
os.environ['CUDA_VISIBLE_DEVICES']='0'



vocab_file = json.load(open(vocab_dir,"r"))

args.num_attr = len(vocab_file)+1
set_seed(args.SEED)
loader = DataLoader(args)
args.vocab_size = loader.vocab_size  
print('vocab_size:',args.vocab_size) 
       
net=LSFF(args)    
net=net.cuda()
net=torch.nn.DataParallel(net)

if args.joint_model!=True:                                                
    args.checkpoint_dir=args.checkpoint_dir+'_'+args.dataset+'_'+args.net
else:
    args.seq_per_image=1                                                                                                          
    if args.vse_loss_type=='contrastive':
        args.checkpoint_dir=args.checkpoint_dir+'_'+args.dataset+'_'+args.net+'_'+str(args.vse_loss_type)+'_'+str(args.vse_margin)
        
    if args.vse_loss_type=='margin':
        args.checkpoint_dir=args.checkpoint_dir+'_'+args.dataset+'_'+args.net+'_'+str(args.vse_loss_type)+'_'+str(args.vse_beta)+'_'+str(args.vse_alpha)


if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir) 

if args.pretrained==True:
    print('==> Resuming from checkpoint..',args.checkpoint_dir)
    assert os.path.isdir(args.checkpoint_dir),'Error: no checkpoint directory !'
    print(os.path.join(args.checkpoint_dir,args.pretrained_model_path))
    net.load_state_dict(torch.load(os.path.join(args.checkpoint_dir,args.pretrained_model_path))['state_dict'])
    best_val_metric=torch.load(os.path.join(args.checkpoint_dir,args.pretrained_model_path))['val_metric']['CIDEr']
    best_epoch=torch.load(os.path.join(args.checkpoint_dir,args.pretrained_model_path))['epoch']
    all_batch_ind=torch.load(os.path.join(args.checkpoint_dir,args.pretrained_model_path))['all_batch_ind'] 

    epoch_start=best_epoch+1
    
    args.self_critical_after=epoch_start                                         
    print('previous best metric:',best_val_metric)
    
       
else:
    print('==> Using model unpretrained..')
    best_val_metric=-1
    epoch_start=0
    all_batch_ind=0

if args.AdamW:
    print("using adamW")
    optimizer = AdamW(net.parameters(),args.learning_rate,warmup=2000)
    epoch_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=args.learning_rate_decay_every,gamma=args.learning_rate_decay_rate) 
    epoch_lr_scheduler.step()                

elif args.RAdam:
    print("using Radam")
    optimizer = RAdam(net.parameters(),args.learning_rate,(args.optim_alpha, args.optim_beta), args.optim_epsilon,weight_decay=args.weight_decay)
    epoch_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=args.learning_rate_decay_every,gamma=args.learning_rate_decay_rate)  
    epoch_lr_scheduler.step()                    

elif args.SGD:
    print("using SGD")
    optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=0.9)
    epoch_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=args.learning_rate_decay_every,gamma=args.learning_rate_decay_rate) 
    epoch_lr_scheduler.step()        
else:
    optimizer=torch.optim.Adam(net.parameters(),args.learning_rate,
                           (args.optim_alpha, args.optim_beta), args.optim_epsilon,
                           weight_decay=args.weight_decay)                    
    epoch_lr_scheduler = lr_scheduler.StepLR(optimizer,
                                             step_size=args.learning_rate_decay_every,
                                             gamma=args.learning_rate_decay_rate)  
    epoch_lr_scheduler.step()                                                     


    
crit = utils.LanguageModelCriterion()
rl_crit =utils.RewardCriterion()                                               


if args.self_critical_after != -1:
    sc_flag = True
    init_scorer(args.cached_tokens)
else:
    sc_flag = False





def train(args):
    global all_batch_ind
    global best_val_metric
    net.train()     
    args.eval_every_steps=len(loader.split_ix['train'])//(args.batch_size)                                                     
    print('eval_every_steps:',args.eval_every_steps)
    end_2=time.time()
    try:
        for epoch in range(epoch_start,epoch_start+args.num_epoches):
            start=time.time()
            batch_idx=0
            
            if epoch-epoch_start>args.learning_rate_decay_start:
                epoch_lr_scheduler.step() 
            learning_rate=epoch_lr_scheduler.get_lr()[0]
            learning_rate1 = optimizer.state_dict()['param_groups'][0]['lr']
            
            
            if epoch > args.scheduled_sampling_start and args.scheduled_sampling_start >= 0:
                frac = (epoch - args.scheduled_sampling_start) // args.scheduled_sampling_increase_every
                args.ss_prob = min(args.scheduled_sampling_increase_prob  * frac, args.scheduled_sampling_max_prob)
                net.module.ss_prob = args.ss_prob
      
            while True:
                data = loader.get_batch('train')
                all_batch_ind=all_batch_ind+1
                batch_idx=batch_idx+1
                
                data['fc_feats']= data['att_feats'].mean(1)                    
                tmp = [data['fc_feats'], data['att_feats'],  data['attr_feats'], data['labels'], data['masks'], data['att_masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]  
                fc_feats, att_feats, attr_feats ,labels, masks, att_masks = tmp        

                

                optimizer.zero_grad()
                
                
                if not sc_flag or epoch<args.self_critical_after: 
                    logits =net(fc_feats,att_feats,attr_feats,labels,att_masks)
                    train_loss= crit(logits, labels[:,1:],masks[:,1:])            
                else:
                    if args.joint_model==False:
                        gen_result, sample_logprobs = net.module.sample(fc_feats, att_feats,attr_feats, att_masks,{'sample_max':0})
                        reward = get_self_critical_reward(net, fc_feats,att_feats,attr_feats,att_masks,data, gen_result,{'sample_max':1,'cider_reward_weight':1,'bleu_reward_weight':0})
                        train_loss =rl_crit(sample_logprobs, gen_result,torch.tensor(reward).float().cuda())
                    else:
                        dis_margin_loss=net.module.cal_retri_loss(model,fc_feats, labels,att_feats, masks,att_masks, args)  
                        
                        gen_result, sample_logprobs = net.module.sample(fc_feats, att_feats, att_masks,{'sample_max':0}) 
                        reward = get_self_critical_reward(net, fc_feats,att_feats,att_masks,data, gen_result, {'sample_max':1,'cider_reward_weight':1,'bleu_reward_weight':0})
                        gen_masks = torch.cat([Variable(gen_result.data.new(gen_result.size(0), 2).fill_(1).float()), (gen_result > 0).float()[:, :-1]], 1)
                        loss_cap = sample_logprobs.cuda() * utils.var_wrapper(-reward.astype('float32')) * (gen_masks[:, 1:].detach().cuda())
                        loss_cap = loss_cap.sum() / gen_masks[:, 1:].cuda().data.float().sum()
            
                        train_loss =dis_margin_loss + loss_cap                                                    
            
                        
                        
                        
                
                if all_batch_ind%20==0:
                    end_1 = time.time()
                    print('epoch:{},ind:{},lr:{},tr_loss:{},ss_prob:{}  ({}/{})   time:{} minites'.format(epoch,all_batch_ind,learning_rate1,train_loss.item(),net.module.ss_prob,batch_idx,len(loader.split_ix['train'])//args.batch_size,str( (end_1-end_2)/60)))                    
                    end_2 = time.time()
                
                    fitlog.add_loss(train_loss.item(),name="loss",step=all_batch_ind//20)
                train_loss.backward()

                utils.clip_gradient(optimizer, args.grad_clip)                 
                optimizer.step()
                torch.cuda.synchronize()


                if (all_batch_ind%(args.eval_every_steps)==0):
                    print('\n\n')
                    print('####################################################')
                    print('Start eval on val dataset...')
                    
                    
                    if  args.beam_size>1:
                        print('using beam search...',args.beam_size)                          
        
                    eval_kwargs=vars(args)
                    eval_kwargs['split']='test'
                    eval_kwargs['dataset']= input_json
                    eval_kwargs['val_images_use']=len(loader.split_ix['test'])
                    
                    
                    
                    
                    
                    val_loss, predictions, lang_stats = eval_utils.eval_split(args,net, crit, loader, eval_kwargs)
                    
                    
                    
                    
                    fitlog.add_metric(lang_stats['CIDEr'].item(),name="cider",step=all_batch_ind//50)
                    
                    print('val_loss'.format(val_loss))
                    print('Eval done!')
                    print('####################################################')
                    print('\n\n')
                    net.train()

                    
                    state={
                            'epoch': epoch,
                            'state_dict': net.state_dict(),
                            'val_metric': lang_stats,
                            'optimizer' : optimizer.state_dict(),
                            'all_batch_ind':all_batch_ind,
                            'learning_rate':learning_rate
                            }
                    
                    if all_batch_ind%(args.eval_every_steps*200)==0:
                        torch.save(state,os.path.join(args.checkpoint_dir,'epoch_{}_batch_idx{}.pkl'.format(epoch,batch_idx)))
                    global best_epoch 
                    if lang_stats['CIDEr']>=best_val_metric:
                        best_epoch=epoch
                        best_val_metric=lang_stats['CIDEr']
                        torch.save(state,os.path.join(args.checkpoint_dir,'best_para.pkl'))
                            
                            
                    print('best  epoch is:{},the best val_metric is:{}'.format(best_epoch, best_val_metric))
                    print('\n\n\n')
                    end = time.time()
                    print('Current epoch cost {} minutes'.format(str( (end-start)/60)))
                    
                    
                if data['bounds']['wrapped']:
                    break
            
            
            
        print('\n\n\n')
        print('best  epoch is:{},the best val_metric is:{}'.format(best_epoch, best_val_metric))
        writer.close()   
        fitlog.finish()
        
    except  KeyboardInterrupt:
        print('The program is terminated, saving the current weight')
        state={
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'val_metric': lang_stats,
                'optimizer' : optimizer.state_dict(),
                'all_batch_ind':all_batch_ind,
                'learning_rate':learning_rate
                }
        torch.save(state,os.path.join(args.checkpoint_dir,'before_KeyboardInterrupt_epoch_{}_batch_idx{}_all_indx{}.pkl'.format(epoch,batch_idx,all_batch_ind)))

def test(args):
    if  args.beam_size>1:
        print('using beam search...',args.beam_size)                          

    eval_kwargs=vars(args)
    eval_kwargs['split']='test'
    eval_kwargs['dataset']= input_json
    eval_kwargs['val_images_use']=len(loader.split_ix['test'])
    
    
    
    
    
    val_loss, predictions, lang_stats = eval_utils.eval_split(args,net, crit, loader, eval_kwargs)
    

def main(args):
#    test(args)
    train(args)


if __name__=='__main__':
    main(args)

            
