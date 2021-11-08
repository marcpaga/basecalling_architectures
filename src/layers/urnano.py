import torch
from torch import nn
import torch.nn.functional as F


class URNetDownBlock(nn.Module):

    def __init__(self, input_size, channel_size, conv_kernel, max_kernel, stride = 1, padding = 'same'):
        super(URNetDownBlock, self).__init__()

        self.conv1 = nn.Conv1d(input_size, channel_size, conv_kernel, stride, padding)
        self.conv2 = nn.Conv1d(channel_size, channel_size, conv_kernel, stride, padding)
        self.batchnorm1 = nn.BatchNorm1d(channel_size)
        self.batchnorm2 = nn.BatchNorm1d(channel_size)
        self.rnn = nn.GRU(channel_size, channel_size)
        self.maxpool = nn.MaxPool1d(max_kernel)

    def forward(self, x):
        """Args:
            x (tensor): shape [batch, channels, len]

        Returns:
            cnv_out (tensor): shape [batch, channels, len]
            rrn_out (tensor): shape [len, batch, channels]
            max_out (tensor): shape [batch, channels, len]
        """
       
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        cnv_out = F.relu(x)

        x = cnv_out.permute(2, 0, 1)
        rnn_out, _ = self.rnn(x)
        
        rnn_out = rnn_out.permute(1, 2, 0)
        max_out = self.maxpool(rnn_out)

        return cnv_out, rnn_out, max_out

class URNetFlatBlock(nn.Module):

    def __init__(self, input_size, channel_size, conv_kernel, stride = 1, padding = 'same'):
        super(URNetFlatBlock, self).__init__()

        self.conv1 = nn.Conv1d(input_size, channel_size, conv_kernel, stride, padding)
        self.conv2 = nn.Conv1d(channel_size, channel_size, conv_kernel, stride, padding)
        self.batchnorm1 = nn.BatchNorm1d(channel_size)
        self.batchnorm2 = nn.BatchNorm1d(channel_size)

    def forward(self, x):
        """Args:
            x (tensor): shape [batch, channels, len]

        Returns:
            x (tensor): shape [batch, channels, len]
        """

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)

        return x

class URNetUpBlock(nn.Module):

    def __init__(self, input_size, channel_size, conv_kernel, up_kernel, up_stride, conv_stride = 1, padding = 'same'):
        super(URNetUpBlock, self).__init__()

        self.conv1 = nn.Conv1d(int(channel_size*3), channel_size, conv_kernel, conv_stride, padding)
        self.conv2 = nn.Conv1d(channel_size, channel_size, conv_kernel, conv_stride, padding)
        self.batchnorm1 = nn.BatchNorm1d(channel_size)
        self.batchnorm2 = nn.BatchNorm1d(channel_size)
        self.upconv = nn.ConvTranspose1d(input_size, channel_size, up_kernel, up_stride)

    def forward(self, x, cnn_in, rnn_in):
        """Args:
            x (tensor): shape [batch, channels, len]
            cnn_in (tensor): shape [batch, channels, len]
            rnn_in (tensor): shape [batch, channels, len]

        Returns:
            out (tensor): shape [batch, channels, len]
        """

        x = self.upconv(x)
        x = torch.cat([x, cnn_in, rnn_in], dim = 1) # concatenate on channel dim
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        out = F.relu(x)

        return out

class URNet(nn.Module):

    def __init__(self, down, flat, up):
        super(URNet, self).__init__()
        """Args:
            down (nn.ModuleList): list of URNetDownBlock(s) to be used
            flat (nn.ModuleList): list of URNetFlatBlock(s) to be used
            up   (nn.ModuleList): list of URNetUpBlock(s) to be used
        """

        self.down = down
        self.flat = flat
        self.up = up


    def forward(self, x):
        """Args:
            x (tensor): shape [batch, channels, len]
            cnn_in (tensor): shape [batch, channels, len]
            rnn_in (tensor): shape [batch, channels, len]

        Returns:
            out (tensor): shape [batch, channels, len]
        """

        cnv_list = list()
        rnn_list = list()
        for layer in self.down:
            cnv_out, rnn_out, x =layer(x)
            cnv_list.append(cnv_out)
            rnn_list.append(rnn_out)


        cnv_list = cnv_list[::-1]
        rnn_list = rnn_list[::-1]
        for layer in self.flat:
            x = layer(x)

        for i, layer in enumerate(self.up):
            x = layer(x, cnv_list[i], rnn_list[i])

        return x