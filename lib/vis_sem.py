import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VisualEncoder(nn.Module):
    def __init__(self, opt):
        super(VisualEncoder, self).__init__()

        self.word_vec_size = opt['word_vec_size']   
        self.combine_emb_dim = opt['jemb_dim']       

        self.conv_dim = opt['fc7_dim']

        self.tanh = nn.Tanh()
        self.conv_proj = nn.Conv2d(self.conv_dim, self.combine_emb_dim, 1, 1)

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.c_linear_v = nn.Linear(1, self.combine_emb_dim)
        self.c_linear_l = nn.Linear(self.word_vec_size, self.combine_emb_dim)
        self.b_proj_linear = nn.Linear(self.combine_emb_dim, 1)

        self.s_linear_v = nn.Linear(self.combine_emb_dim, self.combine_emb_dim)
        self.s_linear_l = nn.Linear(self.word_vec_size, self.combine_emb_dim)
        self.s_linear_sum = nn.Linear(self.combine_emb_dim, 1)

        self.linear_fuse = nn.Sequential(nn.Linear(self.combine_emb_dim, self.word_vec_size * 2),
                                        nn.Tanh())

        self.atten_fuse = nn.Sequential(nn.Linear(self.combine_emb_dim * 2, self.combine_emb_dim),
                                        nn.Linear(self.combine_emb_dim, 1))

    def channel_atention(self, v_feat, l_feat, batch, grid):
        v_feat_pool = self.avg_pooling(v_feat).contiguous().view(batch, 512, -1)  
        vv = self.c_linear_v(v_feat_pool)  ## [N, 512, 512]

        l_feat = self.c_linear_l(l_feat)  ## [N, 1, 512]

        bb = self.tanh(torch.mul(vv, l_feat))  ## [N, 512, 512]
        beta = F.softmax(self.b_proj_linear(bb), dim=1)  ## [N, 512, 1]

        return beta.permute(0, 2, 1)  ## [N, 1, 512]

    def spatial_attention(self, v_feat, l_feat):
        v_feat = self.s_linear_v(v_feat)  ## [N, 49, 512]
        l_feat = self.s_linear_l(l_feat)  ## [N, 49, 512]

        fuse = self.tanh(torch.mul(v_feat, l_feat))
        alpha = F.softmax(self.s_linear_sum(fuse), dim=1)  ## [N, 49, 1]

        return alpha

    def forward(self, conv_feat, phrase_emb):
        batch, grid = conv_feat.size(0), conv_feat.size(2) * conv_feat.size(3)

        v_feat_proj = self.conv_proj(conv_feat)                                         

        v_norm = F.normalize(v_feat_proj, p =2, dim = 1)                                
        l_norm = F.normalize(phrase_emb, p =2 ,dim =1).unsqueeze(1)                    
        
        beta = self.channel_atention(v_norm, l_norm, batch, grid)
        feat0 = torch.mul(beta, v_norm.view(batch, 512, -1).permute(0, 2, 1))          
        
        alpha = self.spatial_attention(feat0, l_norm)                                   
        feat1 = (alpha * feat0).permute(0, 2, 1)
        feat1 = feat1.contiguous().view(batch, 512, 7, 7)                               

        feat = torch.cat([feat1, v_norm], 1)                                             
        feat = feat.transpose(1, 2).transpose(2, 3)
        feat = feat.contiguous().view(batch, grid, 1024)

        l_feat = l_norm.expand(batch, grid, self.word_vec_size)

        fused_feat = self.atten_fuse(torch.mul(feat, l_feat))
        atten = F.softmax(fused_feat.view(batch, grid), dim = 1)                          
        atten3 = atten.unsqueeze(1)
        weighted_visual_feats = torch.bmm(atten3, v_norm.view(batch, grid, -1)).squeeze(1)   

        return  weighted_visual_feats, atten

class RelationEncoder(nn.Module):
    def __init__(self, opt):
        super(RelationEncoder, self).__init__()

        self.fc = nn.Linear(opt['fc7_dim'] * 2 + 5, opt['jemb_dim'])

    def forward(self, cxt_feats, cxt_lfeats, cxt_dif_feats):
        masks = (cxt_lfeats.sum(2) != 0).float()
        batch, num_cxt = cxt_feats.size(0), cxt_feats.size(1)

        cxt_feats = F.normalize(cxt_feats).view(batch * num_cxt, -1)
        cxt_lfeats = F.normalize(cxt_lfeats).view(batch * num_cxt, -1)
        cxt_dif_feats = F.normalize(cxt_dif_feats).view(batch * num_cxt, -1)

        concat_feat = torch.cat([cxt_feats, cxt_lfeats, cxt_dif_feats], 1)
        rel_feats = self.fc(concat_feat).view(batch, num_cxt, -1)

        return rel_feats, masks

class LocationEncoder(nn.Module):
    def __init__(self, opt):
        super(LocationEncoder, self).__init__()

        self.fc = nn.Linear(5 + 25, opt['jemb_dim'])

    def forward(self, lfeats, dif_lfeats):
        lfeats = F.normalize(lfeats)
        dif_lfeats = F.normalize(dif_lfeats)
        concat = torch.cat([lfeats, dif_lfeats], 1)
        output = self.fc(concat)

        return output
