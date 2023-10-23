from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.autograd import Variable
import six
import json
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from torch.nn import TransformerEncoder,TransformerEncoderLayer
from transformer1 import Encoder
import os
from metrics import *
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

#text cnn model
class TextCNN(nn.Module):
    def __init__(self, n_vocab=20000, embed=1024, num_filters=100, filter_sizes=(1,2,3), num_classes=512, dropout=0.1):
    # def __init__(self, n_vocab=12525, embed=1024, num_filters=100, filter_sizes=(1,2,3), num_classes=512, dropout=0.1):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(n_vocab, embed, padding_idx=0)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        #out = self.dropout(out)
        out = self.fc(out)
        
        return out#self.softmax(out)

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

        self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        # im = torch.tensor([[1, 2], [2, 1]])
        # s = torch.tensor([[1, 1], [1, 1]])
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        # cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        aa = cost_s.sum() + cost_im.sum()
        return aa

class CoKEModel(nn.Module):

    def __init__(self,voc_size,emb_size,nhead,nhid,nlayers,dropout,batch_size):
        super(CoKEModel, self).__init__()

        self._emb_size = emb_size
        self._n_layer = nlayers
        self._n_head =nhead
        self._voc_size = voc_size
        self._dropout = dropout
        self._batch_size = batch_size
        self._nhid = nhid
        self.model_type = 'CoKE'
        #self._position_ids =torch.tensor([[0,1,2]])
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device_transformer : ", self._device)
        self.transformer_encoder = Encoder(num_layers=self._n_layer,model_dim=self._emb_size,num_heads=self._n_head,dropout=self._dropout,ffn_dim=self._nhid)

        self.ele_encoder = nn.Embedding(num_embeddings =self._voc_size,embedding_dim = self._emb_size)
        self.post_scale = torch.nn.Parameter(torch.FloatTensor(self._batch_size, 300, self._emb_size), requires_grad=True)
        self.post_bias = torch.nn.Parameter(torch.FloatTensor(self._batch_size, 300,  self._emb_size), requires_grad=True)

        self.dropoutl = torch.nn.Dropout(p = self._dropout)
        self.FC1 = nn.Linear(self._emb_size, self._emb_size)
        # self.FC2 = nn.Linear(self._emb_size,200)
        self.FC2 = nn.Linear(self._emb_size,204)
        self.gelu = nn.GELU()
        self.textcnn = TextCNN()
        self.loacl_global = None
        self.all_loacl_global = None
        self.FC3 = nn.Linear(204,204)
        self.FC4 = nn.Linear(142,204)
        # self.FC5 = nn.Linear(204,512)

        self.label_similarity = torch.nn.Parameter(torch.FloatTensor(200, 200), requires_grad=True)
        
        self.init_weights()
        self.softmax = nn.Softmax(dim=-1)

        self.criterion = ContrastiveLoss(margin=0.02,
                                         measure='cosine',
                                         max_violation=False)


    def layer_norm(self,src):
        begin_norm_axis = len(src.shape)-1
        mean = torch.mean(src,dim=begin_norm_axis,keepdim=True)
        shift_x = src - mean
        variance = torch.mean(shift_x*shift_x,dim=begin_norm_axis,keepdim=True)
        r_stdev = torch.sqrt(1/(variance+1e-12))

        norm_x = shift_x*r_stdev
        return norm_x


    #初始化函数，之后考虑要不要改。
    def init_weights(self):
        initrange = 0.02
        self.ele_encoder.weight.data.normal_(0, 0.02)
        self.FC2.bias.data.zero_()
        self.FC1.bias.data.zero_()
        #self.transformer_encoder.weight.data.normal_(0, 0.02)
        self.FC1.weight.data.normal_(0, 0.02)
        self.FC2.weight.data.normal_(0, 0.02)
        self.FC3.bias.data.zero_()
        self.FC3.weight.data.normal_(0, 0.02)

        self.post_scale.data.fill_(1.)
        self.post_bias.data.fill_(0.)
        
        self.textcnn.embedding.weight.data.normal_(0, 0.02)
        self.textcnn.fc.bias.data.zero_()
        self.textcnn.fc.weight.data.normal_(0, 0.02)
        self.label_similarity.data.fill_(0.)


    def mask_lg(self, src):
        #definite windows size is 10
        loacl_global=torch.ones((self._batch_size,self._n_head,src.size(1),src.size(1)),dtype=torch.bool).to(src.device)
        windows_size = 10
        local_len = 6#3#self._n_head//2
        for i in range(local_len):
            for j in range(src.size(1)):
                for k in range(windows_size):
                    loacl_global[:,i,j,min(max(j-windows_size//2+k, 0), src.size(1)-1)] = False
                    
        for i in range(local_len, self._n_head):
            loacl_global[0,i] = False
            
        loacl_global[:,:,0,:] = False
        
        return loacl_global


    # def forward(self,src,user_info,label_similarity=None):

        # # -----------------------------------------------------------------------------------------
        # # src:(32,200,50)
        # # label_similarity:(204,204)
        # # -----------------------------------------------------------------------------------------
        # src_all_jd = src # (32,200,50)
        # # 当前JD样本
        # src = src[:,:100,:] # (32,100,50)
        # src = src.reshape(-1,50) # (3200,50)
        # src = self.textcnn(src) # (3200,512)
        # src = src.reshape(32,100,self._emb_size) # (32,100,512)
        # src = self.layer_norm(src) # (32,100,512)
        # # 添加cls节点
        # glo_cls = torch.FloatTensor(torch.zeros(src.size(0),1,src.size(2))).to(src.device) # (32,1,512)
        # src = torch.cat([glo_cls, src],dim=1) # (32,101,512)
        #
        #
        # if self.loacl_global == None:
        #     self.loacl_global = self.mask_lg(src).to(src.device) # (32,8,101,101) 固定了以后都是这个
        #
        # pridect, _ = self.transformer_encoder(src, self.loacl_global) # (32,101,512)
        #
        # pridect = pridect.contiguous().view(self._batch_size, -1,self._emb_size) # (32,101,512)
        # pridect = self.FC1(pridect) # (32,101,512)
        # pridect = self.gelu(pridect) # (32,101,512)
        # pridect = self.layer_norm(pridect) # (32,101,512)
        # pridect = pridect * self.post_scale[:,:pridect.size(1),:] # (32,101,512)
        # pridect = pridect + self.post_bias[:,:pridect.size(1),:] # (32,101,512)
        #
        # pridect = self.FC2(pridect) # (32,101,204)
        #
        # # 分别得到cls和token的输出
        # cls_embedd = pridect[:,:1,:] # (32,1,204) 当前样本的cls token
        # token_embedd = pridect[:,1:,:] # (32,100,204) 当前样本的其它token
        #
        # # 加入类别相似度矩阵
        # # 将token的特征转换成(batch, class, token_number)
        # token_embedd = torch.transpose(token_embedd, 1, 2) # (32,204,100)
        # # 将token的特征转换成(batch, class, class)
        # token_embedd = self.FC3(token_embedd) # (32,204,204) 将当前样本变成（204，204）的方阵，方便后续和标签相似度方阵融合
        # # 融入类别相似度矩阵(batch, class, class)与cls矩阵相乘
        # label_similarity = self.softmax(label_similarity) # (204,204)
        # token_embedd = label_similarity + 0.4*token_embedd # (32,204,204)
        # #print("label_similarity : ", label_similarity[:2])
        # #print("token_embedd : ", 0.8*token_embedd[:2])
        # #squeeze()去掉维度为1的维度，比如（5，2，1）-->（5，2）
        # token_embedd = torch.matmul(token_embedd, torch.transpose(cls_embedd, 1,2)).squeeze() # (32,204) 这个token_embedd是指一个样本是204维的向量，它是和全局类别标签交互过的
        #
        #
        # # 计算近邻的损失
        # src_all_jd = src_all_jd.reshape(-1,50) # (6400,50)
        # src_all_jd = self.textcnn(src_all_jd) # (6400,512)
        # src_all_jd = src_all_jd.reshape(32,300,self._emb_size) # (32,200,512)
        # src_all_jd = self.layer_norm(src_all_jd) # (32,200,512)
        # if self.all_loacl_global == None:
        #     self.all_loacl_global = self.mask_lg(src_all_jd).to(src.device) # (32,8,200,200)
        # all_pridect,_ = self.transformer_encoder(src_all_jd, self.all_loacl_global) # (32,200,512)
        # all_pridect = all_pridect.contiguous().view(self._batch_size, -1,self._emb_size) # (32,200,512)
        # all_pridect = self.FC1(all_pridect) # (32,200,512)
        # all_pridect = self.gelu(all_pridect) # (32,200,512)
        # all_pridect = self.layer_norm(all_pridect) # (32,200,512)
        # all_pridect = all_pridect * self.post_scale[:,:all_pridect.size(1),:] # (32,200,512)
        # all_pridect = all_pridect + self.post_bias[:,:all_pridect.size(1),:] # (32,200,512)
        #
        # pridect_1 = self.FC2(all_pridect) # (32,200,204)
        #
        # neighbor_number = all_pridect.size(1)//100 # int:2
        # sims_neighbor_kl = torch.zeros((src.size(0), neighbor_number, neighbor_number)).to(src.device) # (32,2,2)
        # sims_neighbor_euc = torch.zeros((src.size(0), neighbor_number, neighbor_number)).to(src.device) # (32,2,2)
        # for i in range(neighbor_number):
        #     for j in range(neighbor_number):
        #         cls_1 = self.softmax(pridect_1[:,i:(i+1)*100,:].mean(dim=1))
        #         cls_2 = self.softmax(pridect_1[:,j:(j+1)*100,:].mean(dim=1))
        #         all_cls_1 = self.softmax(all_pridect[:,i:(i+1)*100,:].mean(dim=1))
        #         all_cls_2 = self.softmax(all_pridect[:,j:(j+1)*100,:].mean(dim=1))
        #
        #         sims_neighbor_kl[:, i, j] = kl_t_single(cls_1+ 10 ** -6, cls_2+ 10 ** -6) # (32,2,2)
        #         sims_neighbor_euc[:, i, j] = euclidean_t_single(all_cls_1+ 10 ** -6, all_cls_2+ 10 ** -6) # (32,2,2)
        #
        # sims_neighbor_kl = sims_neighbor_kl.reshape(src.size(0), -1)*10 # (32,4)
        # sims_neighbor_euc = sims_neighbor_euc.reshape(src.size(0), -1) # (32,4)
        # sims_neighbor_kl = self.softmax(sims_neighbor_kl) # (32,4)
        # sims_neighbor_euc = self.softmax(sims_neighbor_euc) # (32,4)
        #
        # loss_sims = kl_t(sims_neighbor_kl + 10 ** -6, sims_neighbor_euc + 10 ** -6) # 数值
        # #loss_sims = torch.sum(sims_neighbor_kl) + torch.sum(sims_neighbor_euc)
        #
        # # token_embedd:(32,204)，当前样本的token(和全局标签交互过的)，（III）的输出
        # # cls_embedd:(32,1,204)，当前样本的cls_token
        # # loss_sims:关系一致性的损失值？
        #
        # # -------------------------------------------------------
        # user_embedding = self.FC4(user_info) # 32,204
        # token_embedding = token_embedd # 32,204
        # user_embedding = self.FC5(user_embedding)
        # token_embedding = self.FC5(token_embedding)
        # match_loss = self.criterion(token_embedding, user_embedding)
        # # -------------------------------------------------------
        #
        # a = self.softmax(token_embedd) # 公式5 目前不需要
        # b = self.softmax(cls_embedd)  # 公式4
        # c = loss_sims*100000
        # d = match_loss*0.001 # 用户匹配loss
        #
        #
        # return a, b, c, d

    def forward(self, src, user_info, label_similarity=None):

        # -----------------------------------------------------------------------------------------
        # src:(32,200,50)
        # user_info:(32,142)
        # label_similarity:(204,204)
        # -----------------------------------------------------------------------------------------
        src_all_jd = src  # (32,200,50)
        # 当前JD样本
        src = src[:, :100, :]  # (32,100,50)
        src = src.reshape(-1, 50)  # (3200,50)
        src = self.textcnn(src)  # (3200,512)
        src = src.reshape(32, 100, self._emb_size)  # (32,100,512)
        src = self.layer_norm(src)  # (32,100,512)
        # 添加cls节点
        glo_cls = torch.FloatTensor(torch.zeros(src.size(0), 1, src.size(2))).to(src.device)  # (32,1,512)
        src = torch.cat([glo_cls, src], dim=1)  # (32,101,512)

        if self.loacl_global == None:
            self.loacl_global = self.mask_lg(src).to(src.device)  # (32,8,101,101) 固定了以后都是这个

        pridect, _ = self.transformer_encoder(src, self.loacl_global)  # (32,101,512)
        pridect = pridect.contiguous().view(self._batch_size, -1, self._emb_size)  # (32,101,512)
        pridect = self.FC1(pridect)  # (32,101,512)
        pridect = self.gelu(pridect)  # (32,101,512)
        pridect = self.layer_norm(pridect)  # (32,101,512)
        pridect = pridect * self.post_scale[:, :pridect.size(1), :]  # (32,101,512)
        pridect = pridect + self.post_bias[:, :pridect.size(1), :]  # (32,101,512)
        pridect = self.FC2(pridect)  # (32,101,204)
        # 分别得到当前样本的cls和token的输出
        # cls_embedd:(32,1,204)
        # token_embedd:(32,100,204)
        cls_embedd = pridect[:, :1, :]  # (32,1,204) 当前样本的cls token
        token_embedd = pridect[:, 1:, :]  # (32,100,204) 当前样本的其它token


        # --------------------------------------------------------------------------------------------------------------------------------------------------
        # 引入标签类别相似度矩阵
        P = self.FC3(token_embedd) # (32,100,204)
        cls_embedd_III = self.FC3(cls_embedd) # (32,1,204)
        SLl = P.transpose(1,2)@P # (32,204,204)
        SLg = label_similarity # (204,204)
        SL_hat = SLg + 0.4*SLl # (32,204,204)
        output_III = cls_embedd_III @ SL_hat # (32,1,204)
        output_III = output_III.squeeze() # (32,204)

        # token_embedd = torch.transpose(token_embedd, 1, 2)  # (32,204,100)
        # token_embedd = self.FC3(token_embedd)  # (32,204,204) 将当前样本变成（204，204）的方阵，方便后续和标签相似度方阵融合 （P）
        # label_similarity = self.softmax(label_similarity)  # (204,204)
        # token_embedd = label_similarity + 0.4 * token_embedd  # (32,204,204)
        # token_embedd = torch.matmul(token_embedd, torch.transpose(cls_embedd, 1, 2)).squeeze() # (32,204) 这个token_embedd是指一个样本是204维的向量，它是和全局类别标签交互过的
        # --------------------------------------------------------------------------------------------------------------------------------------------------



        # 计算近邻的损失
        src_all_jd = src_all_jd.reshape(-1, 50)  # (6400,50)
        src_all_jd = self.textcnn(src_all_jd)  # (6400,512)
        src_all_jd = src_all_jd.reshape(32, 200, self._emb_size)  # (32,200,512)
        src_all_jd = self.layer_norm(src_all_jd)  # (32,200,512)
        if self.all_loacl_global == None:
            self.all_loacl_global = self.mask_lg(src_all_jd).to(src.device)  # (32,8,200,200)
        all_pridect, _ = self.transformer_encoder(src_all_jd, self.all_loacl_global)  # (32,200,512)
        all_pridect = all_pridect.contiguous().view(self._batch_size, -1, self._emb_size)  # (32,200,512)
        all_pridect = self.FC1(all_pridect)  # (32,200,512)
        all_pridect = self.gelu(all_pridect)  # (32,200,512)
        all_pridect = self.layer_norm(all_pridect)  # (32,200,512)
        all_pridect = all_pridect * self.post_scale[:, :all_pridect.size(1), :]  # (32,200,512)
        all_pridect = all_pridect + self.post_bias[:, :all_pridect.size(1), :]  # (32,200,512)
        pridect_1 = self.FC2(all_pridect)  # (32,200,204)

        neighbor_number = all_pridect.size(1) // 100  # int:2
        sims_neighbor_kl = torch.zeros((src.size(0), neighbor_number, neighbor_number)).to(src.device)  # (32,2,2)
        sims_neighbor_euc = torch.zeros((src.size(0), neighbor_number, neighbor_number)).to(src.device)  # (32,2,2)
        for i in range(neighbor_number):
            for j in range(neighbor_number):
                cls_1 = self.softmax(pridect_1[:, i:(i + 1) * 100, :].mean(dim=1))
                cls_2 = self.softmax(pridect_1[:, j:(j + 1) * 100, :].mean(dim=1))
                all_cls_1 = self.softmax(all_pridect[:, i:(i + 1) * 100, :].mean(dim=1))
                all_cls_2 = self.softmax(all_pridect[:, j:(j + 1) * 100, :].mean(dim=1))

                sims_neighbor_kl[:, i, j] = kl_t_single(cls_1 + 10 ** -6, cls_2 + 10 ** -6)  # (32,2,2)
                sims_neighbor_euc[:, i, j] = euclidean_t_single(all_cls_1 + 10 ** -6, all_cls_2 + 10 ** -6)  # (32,2,2)
        sims_neighbor_kl = sims_neighbor_kl.reshape(src.size(0), -1) * 10  # (32,4)
        sims_neighbor_euc = sims_neighbor_euc.reshape(src.size(0), -1)  # (32,4)
        sims_neighbor_kl = self.softmax(sims_neighbor_kl)  # (32,4)
        sims_neighbor_euc = self.softmax(sims_neighbor_euc)  # (32,4)

        loss_sims = kl_t(sims_neighbor_kl + 10 ** -6, sims_neighbor_euc + 10 ** -6)  # 数值


        # -------------------------------------------------------
        cls_embedding4match = torch.squeeze(cls_embedd, dim=1)  # (32,204)
        user_embedding4match = self.FC4(user_info) # (32,204)
        cls_embedding_norm = F.normalize(cls_embedding4match, p=2, dim=1) # (32,204)
        user_embedding_norm = F.normalize(user_embedding4match, p=2, dim=1) # (32,204)
        match_user = self.criterion(cls_embedding_norm, user_embedding_norm) # 数值：70左右
        # -------------------------------------------------------

        cls_embedding_SL_hat = self.softmax(output_III)  # 公式5中的g(u_cls)SL_hat    目前不需要
        cls_embedding = self.softmax(cls_embedd)  # 公式4中的g(u_cls)
        relation_consistency_loss = loss_sims * 100000 # 公式6
        match_loss = match_user * 0.01  # 用户匹配loss

        return cls_embedding_SL_hat, cls_embedding, relation_consistency_loss, match_loss


