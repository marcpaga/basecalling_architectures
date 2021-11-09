import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict

class MultiBranch(nn.Module):
    """ Multi branch layer

    Based on: https://github.com/lvxuan96/CATCaller/blob/e23b234a4937207c9ad7e4e69ab812d158fc1242/train/modules.py#L148
    """

    def __init__(self, branches, embed_dim_list):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.embed_dim_list = embed_dim_list

    def forward(self, query, key, value, key_padding_mask=None):
        '''
        :param query/key/value:[batch,seq_len,emb_dim]
        :param key_padding_mask:[batch,1,seq_len]
        :return:
        '''
        _, _, embed_size = query.size()
        assert sum(self.embed_dim_list) == embed_size
        out = []
        attn = None
        start = 0
        for idx, embed_dim in enumerate(self.embed_dim_list):
            branch = self.branches[idx]

            q = query[..., start:start+embed_dim]
            # if key is not None:
            #     assert value is not None
            k, v = key[..., start:start+embed_dim], value[..., start:start+embed_dim]
            start += embed_dim

            if isinstance(branch, torch.nn.MultiheadAttention):  #multiattention forward
                x, attn = branch(query=q, key=k, value=v, mask=key_padding_mask)
            else:
                mask = key_padding_mask
                if mask is not None: #mask[batch,1,seq_len]
                    # print('q:',q.shape)
                    # print('mask:', mask.transpose(-1,-2))
                    q = q.masked_fill(mask.transpose(-1,-2), 0)  #pad need change
                    q = q.transpose(0,1) #[seq_len,batch,emb_dim]
                x = branch(q.contiguous()) #need change unfold
                x = x.transpose(0,1) #[batch,seq_len,emb_dim]
            out.append(x)

        out = torch.cat(out, dim=-1)
        return out, attn

class FFN(nn.Module):
    """ Feed forward layer

    Based on: https://github.com/lvxuan96/CATCaller/blob/e23b234a4937207c9ad7e4e69ab812d158fc1242/train/modules.py#L148
    """
    
    def __init__(self, input_size, hidden_size, dropout=0.15):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.w_2 = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.w_2(x) 
        return output + residual

class DynamicConv1dTBC(nn.Module):
    '''Dynamic lightweight convolution taking T x B x C inputs
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        renorm_padding: re-normalize the filters to ignore the padded part (only the non-padding parts sum up to 1)
        bias: use bias
        conv_bias: bias of the convolution
        query_size: specified when feeding a different input as the query
        in_proj: project the input and generate the filter together
    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`

    Modified from https://github.com/lvxuan96/CATCaller/blob/e23b234a4937207c9ad7e4e69ab812d158fc1242/train/dynamic_convolution.py#L48
    '''
    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1,
                 weight_dropout=0., weight_softmax=False,
                 renorm_padding=False, bias=False, conv_bias=False,
                 query_size=None, in_proj=False, with_linear=False, glu=False):
        super(DynamicConv1dTBC, self).__init__()

        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax
        self.renorm_padding = renorm_padding

        if in_proj:
            self.weight_linear = nn.Linear(self.input_size, self.input_size + num_heads * kernel_size)
        else:
            self.weight_linear = nn.Linear(self.query_size, num_heads * kernel_size, bias=bias)
        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

        if with_linear:
            if glu:
                self.linear1 = nn.Linear(input_size, input_size * 2)
                self.act = nn.GLU()
            else:
                self.linear1 = nn.Linear(input_size, input_size)
                self.act = None
            self.linear2 = nn.Linear(input_size, input_size)

    @property
    def in_proj(self):
        return self.weight_linear.out_features == self.input_size + self.num_heads * self.kernel_size

    def reset_parameters(self):
        self.weight_linear.reset_parameters()
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.)

    def forward(self, x):
        '''Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
            query: use the specified query to predict the conv filters
        '''

        if self.linear1 is not None:
            x = self.linear1(x)
            if self.act is not None:
                x = self.act(x)

        unfold = x.size(0) > 512
        query = x
        
        if unfold:
            output = self._forward_unfolded(x, query)
        else:
            output = self._forward_expanded(x, query)

        if self.conv_bias is not None:
            output = output + self.conv_bias.view(1, 1, -1)
        if self.linear2 is not None:
            output = self.linear2(output)
        return output

    def _forward_unfolded(self, x, query):
        '''The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right.'''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        if self.in_proj:
            proj = self.weight_linear(x)
            x = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H*K).contiguous().view(T*B*H, -1)
        else:
            weight = self.weight_linear(query).view(T*B*H, -1)

        # renorm_padding is only implemented in _forward_expanded
        assert not self.renorm_padding  is not None

        padding_l = self.padding_l
        if K > T and padding_l == K-1:
            weight = weight.narrow(1, K-T, T)
            K, padding_l = T, T-1
        # unfold the input: T x B x C --> T' x B x C x K
        x_unfold = self.unfold1d(x, K, padding_l, 0)
        x_unfold = x_unfold.view(T*B*H, R, K)

        if self.weight_softmax and not self.renorm_padding:
            weight = F.softmax(weight, dim=1)
        weight = weight.narrow(1, 0, K)

        if self.weight_softmax and self.renorm_padding:
            weight = F.softmax(weight, dim=1)

        weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)

        output = torch.bmm(x_unfold, weight.unsqueeze(2))  # T*B*H x R x 1
        output = output.view(T, B, C)
        return output

    def _forward_expanded(self, x, query):
        '''Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.
        '''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size
        if self.in_proj:
            proj = self.weight_linear(x)
            x = proj.narrow(2, 0, self.input_size).contiguous()
            weight = proj.narrow(2, self.input_size, H*K).contiguous().view(T*B*H, -1)
        else:
            weight = self.weight_linear(query).view(T*B*H, -1)

        if not self.renorm_padding:
            if self.weight_softmax:
                weight = F.softmax(weight, dim=1)
            weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)
        weight = weight.narrow(1, 0, K).contiguous()
        weight = weight.view(T, B*H, K).transpose(0, 1)

        x = x.view(T, B*H, R).transpose(0, 1)
        if self.weight_softmax and self.renorm_padding:
            # turn the convolution filters into band matrices
            weight_expanded = weight.new(B*H, T, T+K-1).fill_(float('-inf'))
            weight_expanded.as_strided((B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, self.padding_l, T)
            # normalize the weight over valid positions like self-attention
            weight_expanded = F.softmax(weight_expanded, dim=2)
            weight_expanded = F.dropout(weight_expanded, self.weight_dropout, training=self.training, inplace=False)
        else:
            P = self.padding_l
            # For efficieny, we cut the kernel size and reduce the padding when the kernel is larger than the length
            if K > T and P == K-1:
                weight = weight.narrow(2, K-T, T)
                K, P = T, T-1
            # turn the convolution filters into band matrices
            weight_expanded = weight.new_zeros(B*H, T, T+K-1, requires_grad=False)
            weight_expanded.as_strided((B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
            weight_expanded = weight_expanded.narrow(2, P, T)  # B*H x T x T
        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        return output

    def unfold1d(self, x, kernel_size, padding_l, pad_value=0):
        '''unfold T x B x C to T x B x C x K'''
        if kernel_size > 1:
            T, B, C = x.size()
            x = F.pad(x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value)
            x = x.as_strided((T, B, C, kernel_size), (B*C, C, 1, B*C))
        else:
            x = x.unsqueeze(3)
        return x


class CATCallerEncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, kernel_size, num_heads = 4, channels = 256, dropout = 0.1, weight_softmax = True, weight_dropout = 0.1, with_linear = True, glu = True):
        super(CATCallerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)
        layers = [
            nn.MultiheadAttention(channels, num_heads, dropout),
            DynamicConv1dTBC(
                input_size = channels, 
                kernel_size = kernel_size, 
                padding_l = padding_l, 
                weight_softmax = weight_softmax, 
                weight_dropout= weight_dropout, 
                num_heads = num_heads, 
                with_linear = with_linear,
                glu = glu,
            ),
        ]
        self.slf_attn = MultiBranch(layers, channels)
        self.norm_dropout = nn.Dropout(dropout)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, x_mask):

        ''' 
        :param signal_emb: [batch,seq_len,emb_dim]
        :param src_mask: [batch, seq_len] pad=1
        :return:
        '''

        #sublayer1 MHSA
        residual = x
        input_norm = self.layer_norm(x)
        enc_out, enc_self_attn = self.slf_attn(query=input_norm, key=input_norm, value=input_norm, key_padding_mask=x_mask.bool())  #change parameters for MultiBranch forward
        enc_out = residual + self.norm_dropout(enc_out)

        #sublayer2 FFN
        enc_out = self.ffn(enc_out)

        return enc_out, enc_self_attn
