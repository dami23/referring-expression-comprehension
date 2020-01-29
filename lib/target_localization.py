from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from layers.lang_attention import BiLSTMEncoder, PhraseAttention
from layers.vis_encoder import VisualEncoder, RelationEncoder, LocationEncoder

class Matching(nn.Module):

  def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_drop_out):
    super(Matching, self).__init__()
    self.vis_emb_fc  = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                     nn.BatchNorm1d(jemb_dim),
                                     nn.ReLU(),
                                     nn.Dropout(jemb_drop_out),
                                     nn.Linear(jemb_dim, jemb_dim),
                                     nn.BatchNorm1d(jemb_dim),
                                     )
    self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                     nn.BatchNorm1d(jemb_dim),
                                     nn.ReLU(),
                                     nn.Dropout(jemb_drop_out),
                                     nn.Linear(jemb_dim, jemb_dim),
                                     nn.BatchNorm1d(jemb_dim)
                                     )

  def forward(self, visual_input, lang_input):
    visual_emb = self.vis_emb_fc(visual_input)
    lang_emb = self.lang_emb_fc(lang_input)

    visual_emb_normalized = F.normalize(visual_emb, p=2, dim=1)
    lang_emb_normalized = F.normalize(lang_emb, p=2, dim=1)     

    cossim = torch.sum(visual_emb_normalized * lang_emb_normalized, 1)  # (n, )
    return cossim.view(-1, 1)

class RelationMatching(nn.Module):
  def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_drop_out):
    super(RelationMatching, self).__init__()
    self.lang_dim = lang_dim
    self.vis_emb_fc  = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                     nn.BatchNorm1d(jemb_dim),
                                     nn.ReLU(),
                                     nn.Dropout(jemb_drop_out),
                                     nn.Linear(jemb_dim, jemb_dim),
                                     nn.BatchNorm1d(jemb_dim),
                                     )
    self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                     nn.BatchNorm1d(jemb_dim),
                                     nn.ReLU(),
                                     nn.Dropout(jemb_drop_out),
                                     nn.Linear(jemb_dim, jemb_dim),
                                     nn.BatchNorm1d(jemb_dim)
                                     )
  def forward(self, visual_input, lang_input, masks):
    n, m = visual_input.size(0), visual_input.size(1)
    visual_emb = self.vis_emb_fc(visual_input.view(n*m, -1)) # (n x m, jemb_dim)
    lang_input = lang_input.unsqueeze(1).expand(n, m, self.lang_dim).contiguous() 
    lang_input = lang_input.view(n*m, -1)      # (n x m, lang_dim)
    lang_emb   = self.lang_emb_fc(lang_input)  # (n x m, jemb_dim)

    # l2-normalize
    visual_emb_normalized = F.normalize(visual_emb, p=2, dim=1) 
    lang_emb_normalized   = F.normalize(lang_emb, p=2, dim=1)   

    # compute cossim
    cossim = torch.sum(visual_emb_normalized * lang_emb_normalized, 1)  
    cossim = cossim.view(n, m)   

    cossim = masks * cossim       
    cossim, ixs = torch.max(cossim, 1)

    return cossim.view(-1, 1), ixs


class JointMatching(nn.Module):

  def __init__(self, opt):
    super(JointMatching, self).__init__()
    num_layers = opt['rnn_num_layers']
    hidden_size = opt['rnn_hidden_size']
    num_dirs = 2 if opt['bidirectional'] > 0 else 1

    self.rnn_encoder = BiLSTMEncoder(vocab_size=opt['vocab_size'],
                                  word_embedding_size=opt['word_embedding_size'],
                                  word_vec_size=opt['word_vec_size'],
                                  hidden_size=opt['rnn_hidden_size'],
                                  bidirectional=opt['bidirectional']>0,
                                  input_dropout=opt['word_drop_out'],
                                  lstm_dropout=opt['rnn_drop_out'],
                                  n_layers=opt['rnn_num_layers'],
                                  rnn_type=opt['rnn_type'])

    self.weight_fc = nn.Linear(num_layers * num_dirs * hidden_size, 3)

    self.sub_attn = PhraseAttention(hidden_size * num_dirs * 2)
    self.loc_attn = PhraseAttention(hidden_size * num_dirs * 2)
    self.rel_attn = PhraseAttention(hidden_size * num_dirs * 2)

    self.sub_encoder = VisualEncoder(opt)
    self.sub_matching = Matching(opt['jemb_dim'] * 2, opt['word_vec_size'],
                                 opt['jemb_dim'], opt['jemb_drop_out'])

    self.loc_encoder = LocationEncoder(opt)
    self.loc_matching = Matching(opt['jemb_dim'], opt['word_vec_size'],
                                 opt['jemb_dim'], opt['jemb_drop_out'])

    self.rel_encoder  = RelationEncoder(opt)
    self.rel_matching = RelationMatching(opt['jemb_dim'], opt['word_vec_size'],
                                         opt['jemb_dim'], opt['jemb_drop_out'])


  def forward(self, fc7, lfeats, dif_lfeats, cxt_fc7, cxt_lfeats, cxt_dif_feats, labels, bert_feats):
    context, hidden, embedded = self.rnn_encoder(labels, bert_feats)

    weights = F.softmax(self.weight_fc(hidden), dim=1) 
    
    sub_attn, sub_phrase_emb = self.sub_attn(context, embedded, labels)
    sub_feats, sub_grid_attn = self.sub_encoder(fc7, sub_phrase_emb)
    sub_matching_scores = self.sub_matching(sub_feats, sub_phrase_emb)

    loc_attn, loc_phrase_emb = self.loc_attn(context, embedded, labels)
    loc_feats = self.loc_encoder(lfeats, dif_lfeats)  # (n, 512)
    loc_matching_scores = self.loc_matching(loc_feats, loc_phrase_emb)    

    rel_attn, rel_phrase_emb = self.rel_attn(context, embedded, labels)
    rel_feats, masks = self.rel_encoder(cxt_fc7, cxt_lfeats, cxt_dif_feats)  
    rel_matching_scores, rel_ixs = self.rel_matching(rel_feats, rel_phrase_emb, masks) 

    scores = (weights * torch.cat([sub_matching_scores,
                                   loc_matching_scores,
                                   rel_matching_scores], 1)).sum(1) # (n, )

    return scores, sub_grid_attn, sub_attn, loc_attn, rel_attn, rel_ixs, weights


  def sub_rel_kl(self, sub_attn, rel_attn, input_labels):
    is_not_zero = (input_labels!=0).float()
    sub_attn = Variable(sub_attn.data)  # we only penalize rel_attn
    log_sub_attn = torch.log(sub_attn + 1e-5)
    log_rel_attn = torch.log(rel_attn + 1e-5)

    kl = - sub_attn * (log_sub_attn - log_rel_attn)  # (n, L)
    kl = kl * is_not_zero # (n, L)
    kldiv = kl.sum() / is_not_zero.sum()
    kldiv = torch.exp(kldiv)

    return kldiv
