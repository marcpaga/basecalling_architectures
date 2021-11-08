import torch
from torch import nn
import torch.nn.functional as F

class LightweightConv1d(nn.Module):
    '''Lightweight Convolution assuming the input is TxBxC
    Args:
        input_size (int): # of channels of the input
        kernel_size (int): convolution channels
        padding_l (int): padding to the left when using "same" padding
        num_heads (int): number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout (float): the drop rate of the DropConnect to drop the weight
        weight_softmax (bool): normalize the weight with softmax before the convolution
        bias (bool): use bias
    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`

    Modified from: https://github.com/pytorch/fairseq/blob/28876638114948711fd4bd4e350fdd6809013f1e/fairseq/modules/lightweight_convolution.py
    '''
    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1,
                 weight_dropout=0., weight_softmax=False, bias=False):
        super(LightweightConv1d, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout_module = nn.Dropout(weight_dropout)
        self.weight_softmax = weight_softmax

        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x):
        '''Turn the convolution filters into band matrices and do matrix multiplication.
        This is faster when the sequence is short, but less memory efficient.
        This is not used in the decoder during inference.

        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
        '''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H
        assert R * H == C == self.input_size

        weight = self.weight.view(H, K)
        if self.weight_softmax:
            weight = F.softmax(weight, dim=1, dtype=torch.float32).type_as(weight)
        weight = weight.view(1, H, K).expand(T*B, H, K).contiguous()
        weight = weight.view(T, B*H, K).transpose(0, 1)

        x = x.view(T, B*H, R).transpose(0, 1)
        P = self.padding_l
        if K > T and P == K-1:
            weight = weight.narrow(2, K-T, T)
            K, P = T, T-1
        # turn the convolution filters into band matrices
        weight_expanded = weight.new_zeros(B*H, T, T+K-1, requires_grad=False)
        weight_expanded.as_strided((B*H, T, K), (T*(T+K-1), T+K, 1)).copy_(weight)
        weight_expanded = weight_expanded.narrow(2, P, T)
        weight_expanded = self.weight_dropout_module(weight_expanded)

        output = torch.bmm(weight_expanded, x)
        output = output.transpose(0, 1).contiguous().view(T, B, C)
        
        if self.bias is not None:
            output = output + self.bias.view(1, 1, -1)
        return output