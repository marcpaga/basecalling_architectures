import torch
from torch import nn

class HalcyonCNNBlock(nn.Module):

    def __init__(self, input_channels, num_channels, kernel_size, stride, padding, use_bn):

        self.cnn == nn.Conv1d(input_channels, num_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(num_channels)
        self.relu = nn.ReLU()
        self.use_bn = use_bn

    def forward(self, x):

        x = self.cnn(x)
        x = self.relu(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class HalcyonInceptionBlock(nn.Module):

    def __init__(self, input_channels, num_channels, kernel_sizes, strides, paddings, use_bn, scaler):

        super(HalcyonInceptionBlock, self).__init__()

        self.input_channels = input_channels
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.use_bn = use_bn
        self.scaler = scaler

        self.cnn1 = HalcyonCNNBlock(input_channels, int(num_channels[0] * scaler), kernel_sizes[0], strides[0], paddings[0], use_bn[0])
        self.cnn2 = nn.Sequential(
            HalcyonCNNBlock(input_channels, num_channels[1], 1, strides[1], paddings[1], use_bn[1]), 
            HalcyonCNNBlock(num_channels[1], int(num_channels[1] * scaler), kernel_sizes[1], strides[1], paddings[1], use_bn[1])
        )
        self.cnn3 = nn.Sequential(
            HalcyonCNNBlock(input_channels, num_channels[2], 1, strides[2], paddings[2], use_bn[2]), 
            HalcyonCNNBlock(num_channels[2], int(num_channels[2] * scaler), kernel_sizes[2], strides[2], paddings[2], use_bn[2])
        )
        self.cnn4 = nn.Sequential(
            HalcyonCNNBlock(input_channels, num_channels[3], 1, strides[3], paddings[3], use_bn[3]), 
            HalcyonCNNBlock(num_channels[3], int(num_channels[3] * scaler), kernel_sizes[3], strides[3], paddings[3], use_bn[3])
        )
        self.cnn5 = nn.Sequential(
            nn.AvgPool1d(kernel_sizes[4], strides[4], paddings[4]),
            HalcyonCNNBlock(input_channels, int(num_channels[4] * scaler), 1, strides[4], paddings[4], use_bn[4])
        )
        
    def forward(self, x):
        
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x3 = self.cnn3(x)
        x4 = self.cnn4(x)
        x5 = self.cnn5(x)

        x = torch.cat([x1, x2, x3, x4, x5], dim = 1)
        return x