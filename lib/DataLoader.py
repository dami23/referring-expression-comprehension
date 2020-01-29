from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import cmp_to_key
import os.path as osp
import numpy as np
import h5py
import json
import random
from sklearn.preprocessing import normalize
from loaders.loader import Loader

import torch
from torch.autograd import Variable
from pytorch_pretrained_bert import BertTokenizer, BertModel
torch.manual_seed(1)
from mrcn import inference_no_imdb

def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

class GtMRCNLoader(Loader):
    def __init__(self, data_json, data_h5):
        # parent loader instance
        Loader.__init__(self, data_json, data_h5)

        # prepare attributes
        self.att_to_ix = self.info['att_to_ix']
        self.ix_to_att = {ix: wd for wd, ix in self.att_to_ix.items()}
        self.num_atts = len(self.att_to_ix)
        self.att_to_cnt = self.info['att_to_cnt']

        self.tokenizer = BertTokenizer.from_pretrained('/homeL/mi/vg/bert-large-uncased-vocab.txt')
        self.model = BertModel.from_pretrained('/homeL/mi/vg/bert-large-uncased.tar.gz')
        self.model.eval()
        self.model.to('cuda')

        self.sents = self.to_dict('sentences', 'sent_id')
        self.max_length = self.info['label_length']

        # img_iterators for each split
        self.split_ix = {}
        self.iterators = {}
        for image_id, image in self.Images.items():
            # we use its ref's split (there is assumption that each image only has one split)
            split = self.Refs[image['ref_ids'][0]]['split']
            if split not in self.split_ix:
                self.split_ix[split] = []
                self.iterators[split] = 0
            self.split_ix[split] += [image_id]
        for k, v in self.split_ix.items():
          print('assigned %d images to split %s' % (len(v), k))

    def prepare_mrcn(self, head_feats_dir, args):
        """
        Arguments:
          head_feats_dir: cache/feats/dataset_splitBy/net_imdb_tag, containing all image conv_net feats
          args: imdb_name, net_name, iters, tag
        """
        self.head_feats_dir = head_feats_dir
        self.mrcn = inference_no_imdb.Inference(args)
        assert args.net_name == 'res101'
        self.pool5_dim = 1024
        self.fc7_dim = 2048

    # load different kinds of feats
    def loadFeats(self, Feats):
        # Feats = {feats_name: feats_path}
        self.feats = {}
        self.feat_dim = None
        for feats_name, feats_path in Feats.items():
            if osp.isfile(feats_path):
                self.feats[feats_name] = h5py.File(feats_path, 'r')
                self.feat_dim = self.feats[feats_name]['fc7'].shape[1]
                assert self.feat_dim == self.fc7_dim
                print('FeatLoader loading [%s] from %s [feat_dim %s]' % \
                      (feats_name, feats_path, self.feat_dim))

    # shuffle split
    def shuffle(self, split):
        random.shuffle(self.split_ix[split])

    # reset iterator
    def resetIterator(self, split):
        self.iterators[split] = 0

    # expand list by seq_per_ref, i.e., [a,b], 3 -> [aaabbb]
    def expand_list(self, L, n):
        out = []
        for l in L:
            out += [l] * n
        return out

    def to_dict(self, attr, key):
        dictionary = {}
        for x in self.info[attr]:
            dictionary[x[key]] = x
        return dictionary

    def getBatch(self, split, opt):
        batch_size = opt.get('batch_size', 5)
        seq_per_ref = opt.get('seq_per_ref', 3)
        sample_ratio = opt.get('visual_sample_ratio', 0.3)  
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1                       
        wrapped = False

        batch_image_ids = []
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            image_id = split_ix[ri]
            batch_image_ids += [image_id]

        batch_ref_ids, indexed_tokens, neg_tokens = [], [], []
        batch_pos_ann_ids, batch_pos_sent_ids, batch_pos_pool5, batch_pos_fc7 = [], [], [], []
        pos_indexed_tokens, neg_indexed_tokens, pos_encoded_sums, neg_encoded_sums = [], [], [], []
        batch_neg_ann_ids, batch_neg_sent_ids, batch_neg_pool5, batch_neg_fc7 = [], [], [], []

        for image_id in batch_image_ids:
            ref_ids = self.Images[image_id]['ref_ids']
            batch_ref_ids += self.expand_list(ref_ids, seq_per_ref)

            head, im_info = self.image_to_head(image_id)
            head = Variable(torch.from_numpy(head).cuda())

            image_pos_ann_ids, image_neg_ann_ids = [], []

            for ref_id in ref_ids:
                ref_ann_id = self.Refs[ref_id]['ann_id']

                pos_ann_ids = [ref_ann_id] * seq_per_ref
                pos_sent_ids = self.fetch_sent_ids_by_ref_id(ref_id, seq_per_ref)

                neg_ann_ids, neg_sent_ids = self.sample_neg_ids(ref_ann_id, seq_per_ref, sample_ratio)

                image_pos_ann_ids += pos_ann_ids
                image_neg_ann_ids += neg_ann_ids
                batch_pos_sent_ids += pos_sent_ids
                batch_neg_sent_ids += neg_sent_ids

            pos_ann_boxes = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in image_pos_ann_ids]))
            image_pos_pool5, image_pos_fc7 = self.fetch_grid_feats(pos_ann_boxes, head, im_info)  # (num_pos, k, 7, 7)
            batch_pos_pool5 += [image_pos_pool5]
            batch_pos_fc7   += [image_pos_fc7]
            neg_ann_boxes = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in image_neg_ann_ids]))
            image_neg_pool5, image_neg_fc7 = self.fetch_grid_feats(neg_ann_boxes, head, im_info)  # (num_neg, k, 7, 7)
            batch_neg_pool5 += [image_neg_pool5]
            batch_neg_fc7   += [image_neg_fc7]

            batch_pos_ann_ids += image_pos_ann_ids
            batch_neg_ann_ids += image_neg_ann_ids

        pos_fc7   = torch.cat(batch_pos_fc7, 0); pos_fc7.detach()
        pos_pool5 = torch.cat(batch_pos_pool5, 0); pos_pool5.detach()
        pos_lfeats = self.compute_lfeats(batch_pos_ann_ids)
        pos_dif_lfeats = self.compute_dif_lfeats(batch_pos_ann_ids)
        pos_labels = np.vstack([self.fetch_seq(sent_id) for sent_id in batch_pos_sent_ids])

        for sent_id in batch_pos_sent_ids:
            sent = self.sents[sent_id]['tokens']
            masks = [1] * len(sent)
            while len(sent) < self.max_length:
                sent.append('PAD')
                masks.append(0)
            else:
                if len(sent) == self.max_length:
                    sent = sent
                    masks = masks
                elif len(sent) > self.max_length:
                    sent = sent[:self.max_length]
                    masks = masks[:self.max_length]

            text = ' '.join([x for x in sent])
            tokenized_text = self.tokenizer.tokenize(text)
            token = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            if len(token) > self.max_length:
                token = token[:self.max_length]
            else:
                token = token

            indexed_tokens.append(token)

            tokens_tensor = torch.tensor([token]).to('cuda')
            masks_tensor = torch.tensor([masks]).to('cuda')

            with torch.no_grad():
                encoded_layers, _ = self.model(tokens_tensor, masks_tensor)
            feature_sum = encoded_layers[-1] + encoded_layers[-2] + encoded_layers[-3] + encoded_layers[-4]
            pos_encoded_sums.append(feature_sum)

        pos_indexed_tokens = np.vstack((indexed_tokens))
        batch_pos_encoded_sum = torch.stack((pos_encoded_sums), 1).squeeze(0)

        neg_fc7   = torch.cat(batch_neg_fc7, 0); neg_fc7.detach()
        neg_pool5 = torch.cat(batch_neg_pool5, 0); neg_pool5.detach()
        neg_lfeats = self.compute_lfeats(batch_neg_ann_ids)
        neg_dif_lfeats = self.compute_dif_lfeats(batch_neg_ann_ids)
        neg_labels = np.vstack([self.fetch_seq(sent_id) for sent_id in batch_neg_sent_ids])

        for sent_id in batch_neg_sent_ids:
            sent = self.sents[sent_id]['tokens']
            neg_masks = [1] * len(sent)
            while len(sent) < self.max_length:
                sent.append('PAD')
                neg_masks.append(0)
            else:
                if len(sent) == self.max_length:
                    sent = sent
                    neg_masks = neg_masks
                elif len(sent) > self.max_length:
                    sent = sent[:self.max_length]
                    neg_masks = neg_masks[:self.max_length]

            text = ' '.join([x for x in sent])
            tokenized_text = self.tokenizer.tokenize(text)
            neg_token = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            if len(neg_token) > self.max_length:
                neg_token = neg_token[:self.max_length]
            else:
                neg_token = neg_token

            neg_tokens.append(neg_token)

            neg_tokens_tensor = torch.tensor([neg_token]).to('cuda')
            neg_masks_tensor = torch.tensor([neg_masks]).to('cuda')

            with torch.no_grad():
                encoded_layers, _ = self.model(neg_tokens_tensor, neg_masks_tensor)
            neg_featture_sum = encoded_layers[-1] + encoded_layers[-2] + encoded_layers[-3] + encoded_layers[-4]
            neg_encoded_sums.append(neg_featture_sum)

        neg_indexed_tokens = np.vstack((neg_tokens))
        batch_neg_encoded_sum = torch.stack((neg_encoded_sums), 1).squeeze(0)

        # fetch cxt_fc7 and cxt_lfeats
        pos_cxt_fc7, pos_cxt_lfeats, pos_cxt_dif_feats, pos_cxt_ann_ids = self.fetch_cxt_feats(batch_pos_ann_ids, opt)
        neg_cxt_fc7, neg_cxt_lfeats, neg_cxt_dif_feats, neg_cxt_ann_ids = self.fetch_cxt_feats(batch_neg_ann_ids, opt)
        pos_cxt_fc7 = Variable(torch.from_numpy(pos_cxt_fc7).cuda())
        pos_cxt_lfeats = Variable(torch.from_numpy(pos_cxt_lfeats).cuda())
        pos_cxt_dif_feats = Variable(torch.from_numpy(pos_cxt_dif_feats).cuda())
        neg_cxt_fc7 = Variable(torch.from_numpy(neg_cxt_fc7).cuda())
        neg_cxt_lfeats = Variable(torch.from_numpy(neg_cxt_lfeats).cuda())
        neg_cxt_dif_feats = Variable(torch.from_numpy(neg_cxt_dif_feats).cuda())

        # fetch attributes for batch_pos_ann_ids ONLY
        att_labels, select_ixs = self.fetch_attribute_label(batch_pos_ann_ids)

        pos_lfeats = Variable(torch.from_numpy(pos_lfeats).cuda())
        pos_dif_lfeats = Variable(torch.from_numpy(pos_dif_lfeats).cuda())
        pos_labels = Variable(torch.from_numpy(pos_indexed_tokens).long().cuda())
        # batch_pos_encoded_sum = batch_pos_encoded_sum.cuda()

        neg_lfeats = Variable(torch.from_numpy(neg_lfeats).cuda())
        neg_dif_lfeats = Variable(torch.from_numpy(neg_dif_lfeats).cuda())
        neg_labels = Variable(torch.from_numpy(neg_indexed_tokens).long().cuda())
        # batch_neg_encoded_sum = batch_neg_encoded_sum.cuda()

        # chunk pos_labels and neg_labels using max_len
        max_len = max((pos_labels != 0).sum(1).max().item(),
                      (neg_labels != 0).sum(1).max().item())      ### data[0]
        pos_labels = pos_labels[:, :max_len]
        neg_labels = neg_labels[:, :max_len]

        data = {}
        data['ref_ids'] = batch_ref_ids
        data['ref_ann_ids'] = batch_pos_ann_ids
        data['ref_sent_ids'] = batch_pos_sent_ids
        data['ref_cxt_ann_ids'] = pos_cxt_ann_ids
        data['Feats'] = {'fc7': pos_fc7, 'pool5': pos_pool5, 'lfeats': pos_lfeats, 'dif_lfeats': pos_dif_lfeats,
                         'cxt_fc7': pos_cxt_fc7, 'cxt_lfeats': pos_cxt_lfeats, 'cxt_dif_feats': pos_cxt_dif_feats}
        data['labels'] = pos_labels
        data['bert_feats'] = batch_pos_encoded_sum
        data['neg_ann_ids'] = batch_neg_ann_ids
        data['neg_sent_ids'] = batch_neg_sent_ids
        data['neg_Feats'] = {'fc7': neg_fc7, 'pool5': neg_pool5, 'lfeats': neg_lfeats, 'dif_lfeats': neg_dif_lfeats,
                             'cxt_fc7': neg_cxt_fc7, 'cxt_lfeats': neg_cxt_lfeats, 'cxt_dif_feats': neg_cxt_dif_feats}
        data['neg_labels'] = neg_labels
        data['neg_bert_feats'] = batch_neg_encoded_sum
        data['neg_cxt_ann_ids'] = neg_cxt_ann_ids
        data['att_labels'] = att_labels  # (num_pos_ann_ids, num_atts)
        data['select_ixs'] = select_ixs  # variable size
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}

        return data

    def sample_neg_ids(self, ann_id, seq_per_ref, sample_ratio):
        st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = self.fetch_neighbour_ids(ann_id)

        # neg ann
        neg_ann_ids, neg_sent_ids = [], []
        for k in range(seq_per_ref):
            # neg_ann_id for negative visual representation: mainly from same-type objects
            if len(st_ann_ids) > 0 and np.random.uniform(0, 1, 1) < sample_ratio:
                neg_ann_id = random.choice(st_ann_ids)
            elif len(dt_ann_ids) > 0:
                neg_ann_id = random.choice(dt_ann_ids)
            else:
                neg_ann_id = random.choice(list(self.Anns.keys()))
            neg_ann_ids += [neg_ann_id]
            # neg_ref_id for negative language representations: mainly from same-type "referred" objects
            if len(st_ref_ids) > 0 and np.random.uniform(0, 1, 1) < sample_ratio:
                neg_ref_id = random.choice(st_ref_ids)
            elif len(dt_ref_ids) > 0:
                neg_ref_id = random.choice(dt_ref_ids)
            else:
                neg_ref_id = random.choice(list(self.Refs.keys()))
            neg_sent_id = random.choice(self.Refs[neg_ref_id]['sent_ids'])
            neg_sent_ids += [neg_sent_id]

        return neg_ann_ids, neg_sent_ids

    def fetch_neighbour_ids(self, ref_ann_id):
        ref_ann = self.Anns[ref_ann_id]
        x, y, w, h = ref_ann['box']
        rx, ry = x+w/2, y+h/2

        def compare(ann_id0, ann_id1):
            x, y, w, h = self.Anns[ann_id0]['box']
            ax0, ay0 = x+w/2, y+h/2
            x, y, w, h = self.Anns[ann_id1]['box']
            ax1, ay1 = x+w/2, y+h/2
            # closer --> former
            if (rx-ax0)**2 + (ry-ay0)**2 <= (rx-ax1)**2 + (ry-ay1)**2:
                return -1
            else:
                return 1

        image = self.Images[ref_ann['image_id']]

        ann_ids = list(image['ann_ids'])  # copy in case the raw list is changed
        #ann_ids = sorted(ann_ids, cmp=compare)
        ann_ids = sorted(ann_ids, key=cmp_to_key(compare))

        st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = [], [], [], []
        for ann_id in ann_ids:
            if ann_id != ref_ann_id:
                if self.Anns[ann_id]['category_id'] == ref_ann['category_id']:
                    st_ann_ids += [ann_id]
                    if ann_id in self.annToRef:
                        st_ref_ids += [self.annToRef[ann_id]['ref_id']]
                else:
                    dt_ann_ids += [ann_id]
                    if ann_id in self.annToRef:
                        dt_ref_ids += [self.annToRef[ann_id]['ref_id']]

        return st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids

    def fetch_sent_ids_by_ref_id(self, ref_id, num_sents):
        sent_ids = list(self.Refs[ref_id]['sent_ids'])
        if len(sent_ids) < num_sents:
            append_sent_ids = [random.choice(sent_ids) for _ in range(num_sents - len(sent_ids))]
            sent_ids += append_sent_ids
        else:
            random.shuffle(sent_ids)
            sent_ids = sent_ids[:num_sents]
        assert len(sent_ids) == num_sents
        return sent_ids

    def combine_feats(self, feats0, feats1):
        feats = {}
        for k, v in feats0.items():
            feats[k] = torch.cat([feats0[k], feats1[k]])
        return feats

    def image_to_head(self, image_id):
        feats_h5 = osp.join(self.head_feats_dir, str(image_id)+'.h5')
        feats = h5py.File(feats_h5, 'r')
        head, im_info = feats['head'], feats['im_info']
        return np.array(head), np.array(im_info)

    def fetch_grid_feats(self, boxes, net_conv, im_info):
        pool5, fc7 = self.mrcn.box_to_spatial_fc7(net_conv, im_info, boxes)
        return pool5, fc7

    def compute_lfeats(self, ann_ids):
        lfeats = np.empty((len(ann_ids), 5), dtype=np.float32)
        for ix, ann_id in enumerate(ann_ids):
            ann = self.Anns[ann_id]
            image = self.Images[ann['image_id']]
            x, y, w, h = ann['box']
            ih, iw = image['height'], image['width']
            lfeats[ix] = np.array([[x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)]], np.float32)
        return lfeats

    def compute_dif_lfeats(self, ann_ids, topK=5):
        dif_lfeats = np.zeros((len(ann_ids), 5*topK), dtype=np.float32)
        for i, ref_ann_id in enumerate(ann_ids):
            # reference box
            rbox = self.Anns[ref_ann_id]['box']
            rcx, rcy, rw, rh = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
            # candidate boxes
            _, st_ann_ids, _, _ = self.fetch_neighbour_ids(ref_ann_id)
            for j, cand_ann_id in enumerate(st_ann_ids[:topK]):
                cbox = self.Anns[cand_ann_id]['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                dif_lfeats[i, j*5:(j+1)*5] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])

        return dif_lfeats

    def fetch_cxt_feats(self, ann_ids, opt):
        topK = opt['num_cxt']
        cxt_feats = np.zeros((len(ann_ids), topK, self.fc7_dim), dtype=np.float32)
        cxt_lfeats = np.zeros((len(ann_ids), topK, 5), dtype=np.float32)
        cxt_dif_feats = np.zeros((len(ann_ids), topK, self.fc7_dim), dtype=np.float32)
        cxt_ann_ids = [[-1 for _ in range(topK)] for _ in range(len(ann_ids))] # (#ann_ids, topK)
        for i, ref_ann_id in enumerate(ann_ids):
            # reference box
            rbox = self.Anns[ref_ann_id]['box']
            rcx, rcy, rw, rh = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
            # candidate boxes
            _, st_ann_ids, _, dt_ann_ids = self.fetch_neighbour_ids(ref_ann_id)
            if opt['with_st'] > 0:
                cand_ann_ids = dt_ann_ids + st_ann_ids
            else:
                cand_ann_ids = dt_ann_ids
            cand_ann_ids = cand_ann_ids[:topK]
            for j, cand_ann_id in enumerate(cand_ann_ids):
                cand_ann = self.Anns[cand_ann_id]
                cbox = cand_ann['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                cxt_lfeats[i, j, :] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
                cxt_feats[i, j, :] = self.feats['ann']['fc7'][cand_ann['h5_id'], :]
                cxt_ann_ids[i][j] = cand_ann_id

            for h, cand_ann_id in enumerate(st_ann_ids[:topK]):
                cand_ann = self.Anns[cand_ann_id]
                cxt_dif_feats[i, h, :] = cxt_feats[i, h, :] - self.feats['ann']['fc7'][cand_ann['h5_id'], :]

        vis_dif_feats = []
        for i in range(len(ann_ids)):
            cxt_dif_feat = cxt_dif_feats[i]
            vis_dif_feat = normalize(cxt_dif_feat)
            vis_dif_feats.append(vis_dif_feat)
        cxt_dif_feats = np.vstack(vis_dif_feats).reshape(-1, 5, 2048)

        return cxt_feats, cxt_lfeats, cxt_dif_feats, cxt_ann_ids

    # weights = 1/sqrt(cnt)
    def get_attribute_weights(self, scale=10):
        # return weights for each concept, ordered by cpt_ix
        cnts = [self.att_to_cnt[self.ix_to_att[ix]] for ix in range(self.num_atts)]
        cnts = np.array(cnts)
        weights = 1/cnts**0.5
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        weights = weights * (scale-1) + 1
        return torch.from_numpy(weights).float()

    def fetch_attribute_label(self, ref_ann_ids):
        """Return
        - labels    : Variable float (N, num_atts)
        - select_ixs: Variable long (n, )
        """
        labels = np.zeros((len(ref_ann_ids), self.num_atts))
        select_ixs = []
        for i, ref_ann_id in enumerate(ref_ann_ids):
            ref = self.annToRef[ref_ann_id]
            if len(ref['att_wds']) > 0:
                select_ixs += [i]
                for wd in ref['att_wds']:
                    labels[i, self.att_to_ix[wd]] = 1

        return Variable(torch.from_numpy(labels).float().cuda()), Variable(torch.LongTensor(select_ixs).cuda())

    def decode_attribute_label(self, scores):
        scores = scores.data.cpu().numpy()
        N = scores.shape[0]
        labels = []
        for i in range(N):
            label = []
            score = scores[i]  # (num_atts, )
            for j, sc in enumerate(list(score)):
                label += [(self.ix_to_att[j], sc)]
                labels.append(label)
        return labels

    def getAttributeBatch(self, split):
        wrapped = False
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        ri = self.iterators[split]
        ri_next = ri+1
        if ri_next > max_index:
            ri_next = 0
            wrapped = True
        self.iterators[split] = ri_next
        image_id = split_ix[ri]
        image = self.Images[image_id]

        # fetch head and im_info
        head, im_info = self.image_to_head(image_id)
        head = Variable(torch.from_numpy(head).cuda())

        ref_ids = image['ref_ids']
        ann_ids = [self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]
        ann_boxes  = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in ann_ids]))
        pool5, fc7 = self.fetch_grid_feats(ann_boxes, head, im_info)  # (#ann_ids, k, 7, 7)
        lfeats     = self.compute_lfeats(ann_ids)
        dif_lfeats = self.compute_dif_lfeats(ann_ids)

        lfeats = Variable(torch.from_numpy(lfeats).cuda())
        dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())

        data = {}
        data['image_id'] = image_id
        data['ref_ids'] = ref_ids
        data['ann_ids'] = ann_ids
        data['Feats'] = {'pool5': pool5, 'fc7': fc7, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats}
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}
        return data

    def getSentBatch(self, sent_id, opt):
        # Fetch feats according to the sent_id
        ref = self.sentToRef[sent_id]
        image_id = ref['image_id']
        image = self.Images[image_id]

        head, im_info = self.image_to_head(image_id)
        head = Variable(torch.from_numpy(head).cuda())

        ann_ids = image['ann_ids']
        ann_boxes  = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in ann_ids]))
        pool5, fc7 = self.fetch_grid_feats(ann_boxes, head, im_info) # (#ann_ids, k, 7, 7)
        lfeats     = self.compute_lfeats(ann_ids)
        dif_lfeats = self.compute_dif_lfeats(ann_ids)
        cxt_fc7, cxt_lfeats, cxt_dif_feats, cxt_ann_ids = self.fetch_cxt_feats(ann_ids, opt)

        # labels = np.array([self.fetch_seq(sent_id)]).astype(np.int32)

        tokens, encoded_sum = [], []
        sent = self.sents[sent_id]['tokens']
        masks = [1] * len(sent)
        while len(sent) < self.max_length:
            sent.append('PAD')
            neg_masks.append(0)
        else:
            if len(sent) == self.max_length:
                sent = sent
                masks = masks
            elif len(sent) > self.max_length:
                sent = sent[:self.max_length]
                masks = masks[:self.max_length]

        text = ' '.join([x for x in sent])
        tokenized_text = self.tokenizer.tokenize(text)
        token = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        if len(token) > self.max_length:
            token = token[:self.max_length]
        else:
            token = token

        tokens.append(tokens)

        tokens_tensor = torch.tensor([token]).to('cuda')
        mask_tensor = torch.tensor([masks]).to('cuda')

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, mask_tensor)
        feature_sum = encoded_layers[-1] + encoded_layers[-2] + encoded_layers[-3] + encoded_layers[-4]
        encoded_sum.append(feature_sum)

        batch_tokens = np.vstack((tokens))
        batch_encoded_sum = torch.stack((encoded_sum), 1).squeeze(0)

        lfeats = Variable(torch.from_numpy(lfeats).cuda())
        labels = Variable(torch.from_numpy(batch_tokens).long().cuda())
        
        dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())
        cxt_fc7 = Variable(torch.from_numpy(cxt_fc7).cuda())
        cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())
        cxt_dif_feats = Variable(torch.from_numpy(cxt_dif_feats).cuda())

        data = {}
        data['image_id'] = image_id
        data['ann_ids'] = ann_ids
        data['cxt_ann_ids'] = cxt_ann_ids
        data['Feats']  = {'pool5': pool5, 'fc7': fc7, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                          'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats, 'cxt_dif_feats': cxt_dif_feats}
        data['labels'] = labels
        data['bert_feats'] = batch_encoded_sum
        return data

    def getTestBatch(self, split, opt):
        wrapped = False
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        ri = self.iterators[split]
        ri_next = ri+1
        if ri_next > max_index:
            ri_next = 0
            wrapped = True
        self.iterators[split] = ri_next
        image_id = split_ix[ri]
        image = self.Images[image_id]

        # fetch head and im_info
        head, im_info = self.image_to_head(image_id)
        head = Variable(torch.from_numpy(head).cuda())

        # fetch feats
        ann_ids = image['ann_ids']
        ann_boxes  = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in ann_ids]))
        pool5, fc7 = self.fetch_grid_feats(ann_boxes, head, im_info) # (#ann_ids, k, 7, 7)
        lfeats     = self.compute_lfeats(ann_ids)
        dif_lfeats = self.compute_dif_lfeats(ann_ids)
        cxt_fc7, cxt_lfeats, cxt_dif_feats, cxt_ann_ids = self.fetch_cxt_feats(ann_ids, opt)

        # fetch sents
        sent_ids, gd_ixs, encoded_sum, tokens = [], [], [], []
        for ref_id in image['ref_ids']:
            ref = self.Refs[ref_id]
            for sent_id in ref['sent_ids']:
                sent_ids += [sent_id]
                gd_ixs += [ann_ids.index(ref['ann_id'])]
                # labels = np.vstack([self.fetch_seq(sent_id) for sent_id in sent_ids])

                sent = self.sents[sent_id]['tokens']
                masks = [1] * len(sent)
                while len(sent) < self.max_length:
                    sent.append('PAD')
                    masks.append(0)
                else:
                    if len(sent) == self.max_length:
                        sent = sent
                        masks = masks
                    elif len(sent) > self.max_length:
                        sent = sent[:self.max_length]
                        masks = masks[:self.max_length]

                text = ' '.join([x for x in sent])
                tokenized_text = self.tokenizer.tokenize(text)
                token = self.tokenizer.convert_tokens_to_ids(tokenized_text)

                if len(token) > self.max_length:
                    token = token[:self.max_length]
                else:
                    token = token

                tokens.append(token)

                tokens_tensor = torch.tensor([token]).to('cuda')
                masks_tensor = torch.tensor([masks]).to('cuda')

                with torch.no_grad():
                    encoded_layers, _ = self.model(tokens_tensor, masks_tensor)
                feature_sum = encoded_layers[-1] + encoded_layers[-2] + encoded_layers[-3] + encoded_layers[-4]
                encoded_sum.append(feature_sum)

        batch_tokens = np.vstack((tokens))
        batch_encoded_sum = torch.stack((encoded_sum), 1).squeeze(0)

        # move to Variable
        lfeats = Variable(torch.from_numpy(lfeats).cuda())
        labels = Variable(torch.from_numpy(batch_tokens).long().cuda())
        # batch_encoded_sum = batch_encoded_sum.cuda()
        dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())
        cxt_fc7 = Variable(torch.from_numpy(cxt_fc7).cuda())
        cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())
        cxt_dif_feats = Variable(torch.from_numpy(cxt_dif_feats).cuda())

        data = {}
        data['image_id'] = image_id
        data['ann_ids'] = ann_ids
        data['cxt_ann_ids'] = cxt_ann_ids
        data['sent_ids'] = sent_ids
        data['gd_ixs'] = gd_ixs
        data['Feats']  = {'pool5': pool5, 'fc7': fc7, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                          'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats, 'cxt_dif_feats':cxt_dif_feats}
        data['labels'] = labels
        data['bert_feats'] = batch_encoded_sum
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}
        return data

    def getImageBatch(self, image_id, sent_ids=None, opt={}):
        # fetch head and im_info
        image = self.Images[image_id]
        head, im_info = self.image_to_head(image_id)
        head = Variable(torch.from_numpy(head).cuda())

        # fetch feats
        ann_ids = image['ann_ids']
        ann_boxes  = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in ann_ids]))
        pool5, fc7 = self.fetch_grid_feats(ann_boxes, head, im_info) # (#ann_ids, k, 7, 7)
        lfeats     = self.compute_lfeats(ann_ids)
        dif_lfeats = self.compute_dif_lfeats(ann_ids)
        cxt_fc7, cxt_lfeats, cxt_dif_feats, cxt_ann_ids = self.fetch_cxt_feats(ann_ids, opt)

        # fetch sents
        gd_ixs, tokens, encoded_sum = [], [], []
        if sent_ids is None:
            sent_ids = []
            for ref_id in image['ref_ids']:
                ref = self.Refs[ref_id]
                for sent_id in ref['sent_ids']:
                    sent_ids += [sent_id]
                    gd_ixs += [ann_ids.index(ref['ann_id'])]
        else:
            # given sent_id, we find the gd_ix
            for sent_id in sent_ids:
                ref = self.sentToRef[sent_id]
                gd_ixs += [ann_ids.index(ref['ann_id'])]

                sent = self.sents[sent_id]['tokens']
                masks = [1] * len(sent)
                while len(sent) < self.max_length:
                    sent.append('PAD')
                    masks.append(0)
                else:
                    if len(sent) == self.max_length:
                        sent = sent
                        masks = masks
                    elif len(sent) > self.max_length:
                        sent = sent[:self.max_length]
                        masks = masks[:self.max_length]

                text = ' '.join([x for x in sent])
                tokenized_text = self.tokenizer.tokenize(text)
                token = self.tokenizer.convert_tokens_to_ids(tokenized_text)

                if len(token) > self.max_length:
                    token = token[:self.max_length]
                else:
                    token = token

                tokens.append(token)

                tokens_tensor = torch.tensor([token]).to('cuda')
                masks_tensor = torch.tensor([masks]).to('cuda')

                with torch.no_grad():
                    encoded_layers, _ = self.model(tokens_tensor, masks_tensor)
                feature_sum = encoded_layers[-1] + encoded_layers[-2] + encoded_layers[-3] + encoded_layers[-4]
                encoded_sum.append(feature_sum)

        batch_tokens = np.vstack((tokens))
        batch_encoded_sum = torch.stack((encoded_sum), 1).squeeze(0)
        # labels = np.vstack([self.fetch_seq(sent_id) for sent_id in sent_ids])

        # move to Variable
        lfeats = Variable(torch.from_numpy(lfeats).cuda())
        labels = Variable(torch.from_numpy(batch_tokens).long().cuda())
        # batch_encoded_sum = batch_encoded_sum.cuda()
        dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())
        cxt_fc7 = Variable(torch.from_numpy(cxt_fc7).cuda())
        cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())
        cxt_dif_feats = Variable(torch.from_numpy(cxt_dif_feats).cuda())

        # return data
        data = {}
        data['image_id'] = image_id
        data['ann_ids'] = ann_ids
        data['cxt_ann_ids'] = cxt_ann_ids
        data['sent_ids'] = sent_ids
        data['gd_ixs'] = gd_ixs
        data['Feats']  = {'pool5': pool5, 'fc7': fc7, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats,
                          'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats, 'cxt_dif_feats':cxt_dif_feats}
        data['labels'] = labels
        data['bert_feats'] = batch_encoded_sum

        return data
