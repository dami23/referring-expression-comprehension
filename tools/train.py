from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import numpy as np
import json
import h5py
import time
import random
from pprint import pprint

import _init_paths
from loaders.data_loader import GtMRCNLoader
from layers.joint_match_mymodel import JointMatching
import models.utils as model_utils
import models.eval_mymodel_utils as eval_dets_utils
from crits.max_margin_crit import MaxMarginCriterion
from opt import parse_opt

import torch
import torch.nn as nn
from torch.autograd import Variable

def lossFun(loader, optimizer, model, mm_crit, opt, iter):
    model.train()
    optimizer.zero_grad()

    T = {}
    tic = time.time()
    data = loader.getBatch('train', opt)

    # add [neg_vis, neg_lang]
    Feats = data['Feats']
    labels = data['labels']
    bert_feats = data['bert_feats']

    if opt['visual_rank_weight'] > 0:
        Feats = loader.combine_feats(Feats, data['neg_Feats'])
        labels = torch.cat([labels, data['labels']])
        bert_feats = torch.cat([bert_feats, data['bert_feats']])
        ### (shapes =(2*batch, 2048, 7, 7) )

    if opt['lang_rank_weight'] > 0:
        Feats = loader.combine_feats(Feats, data['Feats'])
        labels = torch.cat([labels, data['neg_labels']])
        bert_feats = torch.cat([bert_feats, data['neg_bert_feats']])
        ### (shapes =(3*batch, 2048, 7, 7) )
    
    T['data'] = time.time()-tic

    # forward
    tic = time.time()
    scores, _, sub_attn, loc_attn, rel_attn, _, _ = model(Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'],
                                                          Feats['cxt_fc7'], Feats['cxt_lfeats'], Feats['cxt_dif_feats'],
                                                          labels, bert_feats)
    loss = mm_crit(scores)
    loss.backward()
    
    torch.nn.utils.clip_grad_value_(model.parameters(), opt['grad_clip'])
    optimizer.step()
    T['model'] = time.time()-tic

    return loss.item(), T, data['bounds']['wrapped']

def main(args):
    opt = vars(args)

    opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
    checkpoint_dir = osp.join(opt['checkpoint_path'], opt['dataset_splitBy'])
    if not osp.isdir(checkpoint_dir): os.makedirs(checkpoint_dir)

    torch.manual_seed(opt['seed'])
    random.seed(opt['seed'])

    data_json = osp.join('/homeL/mi/lieber/MAttNet/tools/cache/prepro', opt['dataset_splitBy'], 'data.json')
    data_h5 = osp.join('/homeL/mi/lieber/MAttNet/tools/cache/prepro', opt['dataset_splitBy'], 'data.h5')
    loader = GtMRCNLoader(data_h5=data_h5, data_json=data_json)

    feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
    head_feats_dir=osp.join('/homeL/mi/vg', opt['dataset_splitBy'], 'mrcn', feats_dir)
    loader.prepare_mrcn(head_feats_dir, args)
    ann_feats = osp.join('/homeL/mi/vg', opt['dataset_splitBy'], 'mrcn',
                         '%s_%s_%s_ann_feats.h5' % (opt['net_name'], opt['imdb_name'], opt['tag']))
    loader.loadFeats({'ann': ann_feats})
    
    opt['vocab_size']= loader.vocab_size   
    opt['fc7_dim']   = loader.fc7_dim       
    opt['pool5_dim'] = loader.pool5_dim    

    model = JointMatching(opt)

    infos = {}
    if opt['start_from'] is not None:
        pass
    iter = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_accuracies = infos.get('val_accuracies', [])
    val_loss_history = infos.get('val_loss_history', {})
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    if opt['load_best_score'] == 1:
        best_val_score = infos.get('best_val_score', None)

    mm_crit = MaxMarginCriterion(opt['visual_rank_weight'], opt['lang_rank_weight'], opt['margin'])

    if torch.cuda.is_available():
        model.cuda()
        mm_crit.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['learning_rate'], betas=(opt['optim_alpha'], opt['optim_beta']),
                                 eps=opt['optim_epsilon'])

    data_time, model_time = 0, 0
    lr = opt['learning_rate']
    best_predictions, best_overall = None, None

    while True:
        loss, T, wrapped = lossFun(loader, optimizer, model, mm_crit, opt, iter)
        data_time += T['data']
        model_time += T['model']

        if iter % opt['losses_log_every'] == 0:
            loss_history[iter] = loss

            log_toc = time.time()
            print('iter[%s](epoch[%s]), train_loss=%.3f, lr=%.2E, data:%.2fs/iter, model:%.2fs/iter' \
                  % (iter, epoch, loss, lr, data_time/opt['losses_log_every'], model_time/opt['losses_log_every']))
            data_time, model_time = 0, 0

        if opt['learning_rate_decay_start'] > 0 and iter > opt['learning_rate_decay_start']:
            frac = (iter - opt['learning_rate_decay_start']) / opt['learning_rate_decay_every']
            decay_factor =  0.1 ** frac
            lr = opt['learning_rate'] * decay_factor
            # update optimizer's learning rate
            model_utils.set_lr(optimizer, lr)

        if iter % opt['save_checkpoint_every'] == 0 or iter == opt['max_iters']:
            val_loss, acc, predictions = eval_dets_utils.eval_split(loader, model, None, 'val', opt)
            val_loss_history[iter] = val_loss
            val_result_history[iter] = {'accuracy': acc}
            val_accuracies += [(iter, acc)]

            print('validation loss: %.2f' % val_loss)
            print('validation acc : %.2f%%\n' % (acc*100.0))

            current_score = acc
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_predictions = predictions

                checkpoint_path = osp.join(checkpoint_dir, opt['id'] + '.pth')
                checkpoint = {}
                checkpoint['model'] = model
                checkpoint['opt'] = opt
                torch.save(checkpoint, checkpoint_path)
                print('model saved to %s' % checkpoint_path)

            infos['iter'] = iter
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['loss_history'] = loss_history
            infos['val_accuracies'] = val_accuracies
            infos['val_loss_history'] = val_loss_history
            infos['best_val_score'] = best_val_score
            infos['best_predictions'] = predictions if best_predictions is None else best_predictions
            infos['opt'] = opt
            infos['val_result_history'] = val_result_history
            infos['word_to_ix'] = loader.word_to_ix
            infos['att_to_ix'] = loader.att_to_ix
            with open(osp.join(checkpoint_dir, opt['id']+'.json'), 'wb') as io:
                # json.dump(infos, io)
                io.write((json.dumps(infos).encode("utf-8")))
                io.close()

        iter += 1
        if wrapped:
            epoch += 1
        if iter >= opt['max_iters'] and opt['max_iters'] > 0:
            break

if __name__ == '__main__':
    args = parse_opt()
    main(args)
