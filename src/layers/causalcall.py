"""Implementations of the layers used in the CausalCall model
"""

from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

class CausalCallConvBlock(nn.Module):
    """Calsual convolutional block from CausalCall

    Based on: https://www.frontiersin.org/articles/10.3389/fgene.2019.01332/full#h3 
    """


    def __init__(self, 
        kernel_size, 
        num_channels,
        first_channel,
        dilation):

        super(CausalCallConvBlock, self).__init__()

        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.first_channel = first_channel
        self.dilation = dilation

        # divide input channel for the second conv2 and the conv_res by 2
        # because the glu operation divides the number of channels in two
        self.conv1 = weight_norm(CausalConv1d(self.first_channel, self.num_channels, self.kernel_size, 1, self.dilation))
        self.conv2 = weight_norm(CausalConv1d(int(self.num_channels/2), self.num_channels, self.kernel_size, 1, self.dilation))
        self.conv_res = nn.Conv1d(self.first_channel, int(self.num_channels/2), 1, 1)

    def forward(self, x):
        """Forward through the network
        Args:
            x (tensor): input with shape [batch, channels, timesteps]
        """
        residual = x
        xb = self.conv1(x)
        xb = F.glu(xb, dim = 1) # dim=1 is the channels dimensions
        xb = self.conv2(xb)
        xb = F.glu(xb, dim = 1)

        if self.first_channel != self.num_channels:
            residual = self.conv_res(residual)

        xb += residual

        xb = F.relu(xb)
        return xb
