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
# from torch.nn import TransformerEncoder,TransformerEncoderLayer
from transformer1 import Encoder
import os
from metrics import *


# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


# text cnn model
class TextCNN(nn.Module):
    def __init__(self, n_vocab=20000, embed=1024, num_filters=100, filter_sizes=(1, 2, 3), num_classes=512,
                 dropout=0.1):
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
        # out = self.dropout(out)
        out = self.fc(out)

        return out  # self.softmax(out)


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
        cost_s = (0.1 + scores - d1).clamp(min=0)
        # cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (0.1 + scores - d2).clamp(min=0)
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

    def __init__(self, voc_size, emb_size, nhead, nhid, nlayers, dropout, batch_size):
        super(CoKEModel, self).__init__()

        self._emb_size = emb_size
        self._n_layer = nlayers
        self._n_head = nhead
        self._voc_size = voc_size
        self._dropout = dropout
        self._batch_size = batch_size
        self._nhid = nhid
        self.model_type = 'CoKE'
        # self._position_ids =torch.tensor([[0,1,2]])
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device_transformer : ", self._device)
        self.transformer_encoder = Encoder(num_layers=self._n_layer, model_dim=self._emb_size, num_heads=self._n_head,
                                           dropout=self._dropout, ffn_dim=self._nhid)

        self.ele_encoder = nn.Embedding(num_embeddings=self._voc_size, embedding_dim=self._emb_size)
        self.post_scale = torch.nn.Parameter(torch.FloatTensor(self._batch_size, 300, self._emb_size),
                                             requires_grad=True)
        self.post_bias = torch.nn.Parameter(torch.FloatTensor(self._batch_size, 300, self._emb_size),
                                            requires_grad=True)

        self.dropoutl = torch.nn.Dropout(p=self._dropout)
        self.FC1 = nn.Linear(self._emb_size, self._emb_size)
        # self.FC2 = nn.Linear(self._emb_size,200)
        self.FC2 = nn.Linear(self._emb_size, 204)
        self.gelu = nn.GELU()
        self.textcnn = TextCNN()
        self.loacl_global = None
        self.all_loacl_global = None
        self.FC3 = nn.Linear(100, 204)
        self.FC4 = nn.Linear(142, 204)
        # self.FC_jingpai = nn.Linear(716, 1)
        self.label_similarity = torch.nn.Parameter(torch.FloatTensor(200, 200), requires_grad=True)

        # self.fc1 = nn.Linear(512, 204)
        # self.fc2 = nn.Linear(204, 204)
        # self.fc3 = nn.Linear(512, 512)
        # self.fc4 = nn.Linear(512, 512)
        # self.fc5 = nn.Linear(204, 512)
        # self.fc6 = nn.Linear(512, 1)

        self.fc1 = nn.Linear(512, 204)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(204, 204)
        self.fc4 = nn.Linear(204, 204)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 1)

        self.init_weights()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = ContrastiveLoss(margin=0.2,
                                         measure='cosine',
                                         max_violation=False)

    def freeze_train(self):
        freeze_blocks = [self.transformer_encoder, self.ele_encoder, self.post_bias, self.post_scale, self.FC1,
                         self.FC2, self.FC3, self.FC4, self.textcnn, self.label_similarity]

        for module in freeze_blocks:
            try:
                for param in module.parameters():
                    param.requires_grad = False
            except:
                module.requires_grad = False

    def layer_norm(self, src):
        begin_norm_axis = len(src.shape) - 1
        mean = torch.mean(src, dim=begin_norm_axis, keepdim=True)
        shift_x = src - mean
        variance = torch.mean(shift_x * shift_x, dim=begin_norm_axis, keepdim=True)
        r_stdev = torch.sqrt(1 / (variance + 1e-12))

        norm_x = shift_x * r_stdev
        return norm_x

    # 初始化函数，之后考虑要不要改。
    def init_weights(self):
        initrange = 0.02
        self.ele_encoder.weight.data.normal_(0, 0.02)
        self.FC2.bias.data.zero_()
        self.FC1.bias.data.zero_()
        # self.transformer_encoder.weight.data.normal_(0, 0.02)
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
        # definite windows size is 10
        loacl_global = torch.ones((self._batch_size, self._n_head, src.size(1), src.size(1)), dtype=torch.bool).to(
            src.device)
        windows_size = 10
        local_len = 6  # 3#self._n_head//2
        for i in range(local_len):
            for j in range(src.size(1)):
                for k in range(windows_size):
                    loacl_global[:, i, j, min(max(j - windows_size // 2 + k, 0), src.size(1) - 1)] = False

        for i in range(local_len, self._n_head):
            loacl_global[0, i] = False

        loacl_global[:, :, 0, :] = False

        return loacl_global

    def forward(self, src, user_feature, epoch=None, label_similarity=None):

        # -----------------------------------------------------------------------------------------
        # src=train_text:(32,200,50)
        # label_similarity:(200,200)
        # -----------------------------------------------------------------------------------------

        # print(src)
        # src = self.ele_encoder(src)
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

        # pridect = pridect[:,:101,:]
        # print("cls_embedd : ", pridect[:,:1,:])
        # pridect = self.FC2(pridect) # (32,101,200)--32，101，204
        # pridect = l2norm(pridect, dim=-1)

        # 分别得到当前样本cls和token的输出
        cls_embedd = pridect[:, :1, :]  # (32,1,512)
        token_embedd = pridect[:, 1:, :]  # (32,100,512)

        # ---------------------------------------------------------------
        # cls_embedd = cls_embedd.squeeze()
        # all_feature = torch.cat([cls_embedd, user_feature],dim=1)
        # all_feature = self.FC_jingpai(all_feature)
        # all_feature = self.sigmoid(all_feature)
        # ---------------------------------------------------------------


        jd_embedding = cls_embedd.squeeze() # (32,512)
        user_embedding = user_feature # (32,204)
        # jd_embedding = self.fc1(jd_embedding)

        # q = self.fc2(user_embedding) # (32,204)
        # k = self.fc3(jd_embedding) # (32,512)
        # v = self.fc4(jd_embedding) # (32,512)
        # q = q.unsqueeze(-1) # (32,204,1)
        # k = k.unsqueeze(-1) # (32,512,1)
        # v = v.unsqueeze(-1) # (32,512,1)
        #
        # attn = (q @ k.transpose(-2, -1)) * ((1 // 1) ** -0.5) # (32,204,512)
        # weights = self.softmax(attn) # (32,204,512)
        #
        #
        # output = (weights @ v).squeeze() # (32,204)
        #
        # output = self.fc5(output) # (32,512)
        # output = nn.ReLU()(output)
        # output = self.fc6(output) # (32,1)
        # output = self.sigmoid(output)
        #
        # return output, weights

        q = self.fc2(jd_embedding)  # (32,512)
        k = self.fc3(user_embedding)  # (32,204)
        v = self.fc4(user_embedding)  # (32,204)
        q = q.unsqueeze(-1)  # (32,512,1)
        k = k.unsqueeze(-1)  # (32,204,1)
        v = v.unsqueeze(-1)  # (32,204,1)

        attn = (q @ k.transpose(-2, -1)) * ((1 // 1) ** -0.5)  # (32,512,204)
        weights = self.softmax(attn)  # (32,512,204)

        cross = weights.cpu().detach().numpy()
        np.save('./cross.npy', cross)

        output = (weights @ v).squeeze()  # (32,512)

        output = self.fc5(output)  # (32,512)
        output = nn.ReLU()(output)
        output = self.fc6(output)  # (32,1)
        output = self.sigmoid(output)

        return output, weights



