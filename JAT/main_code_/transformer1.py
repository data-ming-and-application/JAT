import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    #"""Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None, attn_mask=None, bias=None):
        """前向传播.
        Args:
                q: Queries张量，形状为[B, L_q, D_q]
                k: Keys张量，形状为[B, L_k, D_k]
                v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
                scale: 缩放因子，一个浮点标量
                attn_mask: Masking张量，形状为[B, L_q, L_k]
        Returns:
                上下文张量和attetention张量
        """

        attention = q
        if scale:
                attention = attention * scale
        attention = torch.matmul(q, k.transpose(-1, -2))
        #if attention.size(2) == 101:
            #print("attention : ", attention.shape, attention[:][0][0][:])
        if attn_mask:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
                
        if bias != None:
            #print("bias : ", bias.shape)
            attention = attention.masked_fill(bias, -10 ** 8)#float('-inf'))
            
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        #print(attention.shape)
        # 和V做点积
        #print('v',v.shape)
        context = torch.matmul(attention, v)
        #print(context.shape)
        return context, attention

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=256, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads*2
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        #self.weight_init()
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim*2, model_dim)
        self.dropout = nn.Dropout(0.)
                # multi-head attention之后需要做layer norm
        self.layer_norm = LayerNorm(model_dim)
        self.weight_init()
    def weight_init(self):
        initrange = 0.02
        self.linear_k.weight.data.normal_(0, initrange)
        self.linear_v.weight.data.normal_(0, initrange)
        self.linear_q.weight.data.normal_(0, initrange)
        self.linear_k.bias.data.zero_()
        self.linear_v.bias.data.zero_()
        self.linear_q.bias.data.zero_()
        self.linear_final.weight.data.normal_(0,initrange)
        self.linear_final.bias.data.zero_()
    def forward(self, key, value, query, attn_mask=None, bias=None):
                # 残差连接
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        #batch_size = key.size(0)
        batch_size = 32
        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size ,-1, num_heads,  dim_per_head)
        value = value.view(batch_size ,-1, num_heads,  dim_per_head)
        query = query.view(batch_size ,-1, num_heads,  dim_per_head)
        #print('key',key.shape, 'value',value.shape, 'query',query.shape)
        key = key.transpose(2,1)
        value = value.transpose(2,1)
        query = query.transpose(2,1)
        #print('key',key.shape)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask, bias=bias)

        # concat heads
        context = context.transpose(2,1)

        context = context.reshape(batch_size, -1, dim_per_head * num_heads)
        output = self.linear_final(context)
        return output, attention


def residual(sublayer_fn, x):
    return sublayer_fn(x) + x
class LayerNorm(nn.Module):
    """实现LayerNorm。其实PyTorch已经实现啦，见nn.LayerNorm。"""

    def __init__(self, features, epsilon=1e-12):
        """Init.
        Args:
            features: 就是模型的维度。论文默认512
            epsilon: 一个很小的数，防止数值计算的除0错误
        """
        super(LayerNorm, self).__init__()
        # alpha
        self.gamma = nn.Parameter(torch.ones(features))
        # # beta
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon
        self._batch_size = 32
        self._emb_size = features
        self.pre_scale = torch.nn.Parameter(torch.FloatTensor(self._batch_size, 300, self._emb_size), requires_grad=True)
        self.pre_bias = torch.nn.Parameter(torch.FloatTensor(self._batch_size, 300, self._emb_size), requires_grad=True)
        self.weight_init()
    def weight_init(self):
        self.pre_scale.data.fill_(1.)
        self.pre_bias.data.fill_(0.)
    def forward(self, src):
        # """前向传播.
        # Args:
        #     x: 输入序列张量，形状为[B, L, D]
        # """
        # # 根据公式进行归一化
        # # 在X的最后一个维度求均值，最后一个维度就是模型的维度
        #mean = x.mean(-1, keepdim=True)
        # # 在X的最后一个维度求方差，最后一个维度就是模型的维度
        #std = x.std(-1, keepdim=True)
        #return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
        begin_norm_axis = len(src.shape)-1
        #print('transformer1',begin_norm_axis)
        mean = torch.mean(src, dim=begin_norm_axis, keepdim=True)
        shift_x = src - mean
        variance = torch.mean(shift_x * shift_x, dim=begin_norm_axis, keepdim=True)
        r_stdev = torch.sqrt(1 / (variance + self.epsilon))
        norm_x = shift_x * r_stdev
        norm_x = norm_x*self.pre_scale[:,:norm_x.size(1),:]
        norm_x = norm_x+self.pre_bias[:,:norm_x.size(1),:]
        return norm_x

class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=256, ffn_dim=512, dropout=0.):
        super(PositionalWiseFeedForward, self).__init__()
        self.f1 = nn.Linear(model_dim,ffn_dim)
        self.gelu = nn.GELU()
        self.f2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(0.)
        self.weight_init()
    def weight_init(self):
        self.f1.weight.data.normal_(0, 0.02)
        self.f2.weight.data.normal_(0, 0.02)
        self.f1.bias.data.zero_()
        self.f1.bias.data.zero_()
    def forward(self, x):
        #output = x.transpose(1, 2)
        #print('output',output.shape)
        output = self.f1(x)
        output = self.gelu(output)
        output = self.dropout(output)
        output = self.f2(output)
        #output = self.dropout(output)

        # add residual and norm layer


        return output

class EncoderLayer(nn.Module):
        # """Encoder的一层。"""
    def __init__(self, model_dim=256, num_heads=4, ffn_dim=512, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm2 = LayerNorm(model_dim)
        self.layer_norm1 = LayerNorm(model_dim)
    def forward(self, inputs, attn_mask=None, bias=None):
        residual = inputs
        inputs = self.layer_norm1(inputs)
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, bias=bias)
        context = self.dropout(residual+context)
        residual = context
        # feed forward network
        context = self.layer_norm2(context)
        output = self.feed_forward(context)
        #output = context+output
        output = self.dropout(output+residual)
        return output, attention


class Encoder(nn.Module):
        #"""多层EncoderLayer组成Encoder。"""

    def __init__(self,
               num_layers=12,
               model_dim=256,
               num_heads=4,
               ffn_dim=512,
               dropout=0.1):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.layer_norm =LayerNorm(model_dim)
    def forward(self, src, bias=None):
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(src, bias=bias)
            #print('encoderlayer',output.shape)
            attentions.append(attention)
        output = self.layer_norm(output)
        return output, attentions
