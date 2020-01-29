from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import json
import numpy as np
import h5py
import time
from pprint import pprint
import argparse

import _init_paths
from layers.joint_match_sem_loc_rel import JointMatching
from loaders.gt_mrcn_loader import GtMRCNLoader
#import models.eval_easy_utils as eval_utils
import models.eval_sem_loc_rel as eval_dets_utils

import torch
import torch.nn as nn

def load_model(checkpoint_path, opt):
  tic = time.time()
  model = JointMatching(opt)
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model'].state_dict())
  model.eval()
  model.cuda()
  print('model loaded in %.2f seconds' % (time.time()-tic))
  return model

def evaluate(params):
  data_json = osp.join('/home/mi/MAttNet/tools/cache/prepro', params['dataset_splitBy'], 'data.json')
  data_h5 = osp.join('/home/mi/MAttNet/tools/cache/prepro', params['dataset_splitBy'], 'data.h5')
  loader = GtMRCNLoader(data_h5=data_h5, data_json=data_json)

  # load mode info
  model_prefix = osp.join('output', params['dataset_splitBy'], params['id'])
  infos = json.load(open(model_prefix+'.json'))
  model_opt = infos['opt']
  model_path = model_prefix + '.pth'
  model = load_model(model_path, model_opt)

  feats_dir = '%s_%s_%s' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag'])
  args.imdb_name = model_opt['imdb_name']
  args.net_name = model_opt['net_name']
  args.tag = model_opt['tag']
  args.iters = model_opt['iters']
  loader.prepare_mrcn(head_feats_dir=osp.join('/home/mi/', params['dataset_splitBy'], 'mrcn', feats_dir),
                      args=args)
  ann_feats = osp.join('/home/mi/', params['dataset_splitBy'], 'mrcn',
                       '%s_%s_%s_ann_feats.h5' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag']))
  loader.loadFeats({'ann': ann_feats})

  assert model_opt['dataset'] == params['dataset']
  assert model_opt['splitBy'] == params['splitBy']

  split = params['split']
  model_opt['num_sents'] = params['num_sents']
  model_opt['verbose'] = params['verbose']
  crit = None
  val_loss, acc, predictions = eval_dets_utils.eval_split(loader, model, crit, split, model_opt)
  print('Comprehension on %s\'s %s (%s sents) is %.2f%%' % \
        (params['dataset_splitBy'], params['split'], len(predictions), acc*100.))

  out_dir = osp.join('cache', 'results', params['dataset_splitBy'], 'easy')
  if not osp.isdir(out_dir):
    os.makedirs(out_dir)
  out_file = osp.join(out_dir, params['id']+'_'+params['split']+'.json')
  with open(out_file, 'w') as of:
    json.dump({'predictions': predictions, 'acc': acc}, of)

  f = open('output/easy_results.txt', 'a')
  f.write('[%s][%s], id[%s]\'s acc is %.2f%%\n' % \
          (params['dataset_splitBy'], params['split'], params['id'], acc*100.0))


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='refcoco', help='dataset name: refclef, refcoco, refcoco+, refcocog')
  parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
  parser.add_argument('--split', type=str, default='testA', help='split: testAB or val, etc')
  parser.add_argument('--id', type=str, default='0', help='model id name')
  parser.add_argument('--num_sents', type=int, default=-1, help='how many sentences to use when periodically evaluating the loss? (-1=all)')
  parser.add_argument('--verbose', type=int, default=1, help='if we want to print the testing progress')
  args = parser.parse_args()
  params = vars(args)

  params['dataset_splitBy'] = params['dataset'] + '_' + params['splitBy']
  evaluate(params)
