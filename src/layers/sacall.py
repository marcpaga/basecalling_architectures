'''Code adapted from: https://github.com/huangnengCSU/SACall-basecaller/blob/ee18a1e5e857810166bd73d05f672a25c9a61a8b/transformer/modules.py#L111
'''

import math
import torch
from torch import nn
import torch.nn.functional as F

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -float('inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class multiheadattention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(multiheadattention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        # nn.init.xavier_normal_(self.w_q.weight)
        # nn.init.xavier_normal_(self.w_k.weight)
        # nn.init.xavier_normal_(self.w_v.weight)
        # nn.init.xavier_normal_(self.fc.weight)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query = self.w_q(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        output, self.attn = attention(query=query, key=key, value=value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        output = output.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        output = self.fc(output)

        return output, self.attn


class FFN(nn.Module):
    # feedforward layer
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.w_2 = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.w_2(inter)
        return output + x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = multiheadattention(h=n_head, d_model=d_model, dropout=dropout)
        self.ffn = FFN(input_size=d_model, hidden_size=d_ff, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_dropout = nn.Dropout(dropout)

    def forward(self, signal_emb, src_mask = None):
        input = signal_emb
        input_norm = self.layer_norm(signal_emb)
        enc_out, _ = self.slf_attn(input_norm, input_norm, input_norm, src_mask)
        enc_out = input + self.norm_dropout(enc_out)

        enc_out = self.ffn(enc_out)

        return enc_out