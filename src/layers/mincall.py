"""Implementations of the layers used in the MinCall model
"""

from torch import nn

class MinCallConvBlock(nn.Module):
    """Convolutional residual block as described in Miculinic et al., 2019
        https://arxiv.org/pdf/1904.10337.pdf
    """

    def __init__(self, 
        kernel_size, 
        num_channels,
        first_channel,
        padding = 'same'):

        super(MinCallConvBlock, self).__init__()

        self.kernel_size = kernel_size
        self.padding = padding
        self.num_channels = num_channels
        self.first_channel = first_channel

        self.bn1 = nn.BatchNorm1d(self.first_channel)
        self.conv1 = nn.Conv1d(self.first_channel, self.num_channels, self.kernel_size, 1, self.padding)
        self.elu1 = nn.ELU()
        self.bn2 = nn.BatchNorm1d(self.num_channels)
        self.conv2 = nn.Conv1d(self.num_channels, self.num_channels, self.kernel_size, 1, self.padding)
        self.elu2 = nn.ELU()

    def forward(self, x):
        """Forward through the network
        Args:
            x (tensor): input with shape [batch, channels, timesteps]
        """
        residual = x
        xb = self.bn1(x)
        xb = self.elu1(xb)
        xb = self.conv1(xb)
        xb = self.bn2(xb)
        xb = self.elu2(xb)
        xb = self.conv2(xb)
        xb += residual

        return xb