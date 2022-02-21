import torch
from torch import nn

class HalcyonCNNBlock(nn.Module):

    def __init__(self, input_channels, num_channels, kernel_size, stride, padding, use_bn, input_length = None):
        super(HalcyonCNNBlock, self).__init__()

        if padding == 'same' and input_length is not None:
            padding = self._calculate_padding()
        else:
            try:
                self.cnn = nn.Conv1d(input_channels, num_channels, kernel_size, stride, padding)
            except:
                self.cnn = nn.Conv1d(input_channels, num_channels, kernel_size, stride, 'valid')

        self.input_length = input_length
        self.input_channels = input_channels
        self.num_channels = num_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.bn = nn.BatchNorm1d(num_channels)
        self.relu = nn.ReLU()
        self.use_bn = use_bn

    def forward(self, x):

        if self.input_length is None and self.padding != 'valid':
            self.input_length = x.shape[2]
            self.padding = self._calculate_padding(self.input_length, self.stride, self.kernel_size)
            self.cnn = nn.Conv1d(self.input_channels, self.num_channels, self.kernel_size, self.stride, self.padding).to(x.device)

        x = self.cnn(x)
        x = self.relu(x)
        if self.use_bn:
            x = self.bn(x)
        return x

    def _calculate_padding(self, l, s, k):

        return int((l*s - s + k - l)/2)

class HalcyonInceptionBlock(nn.Module):

    def __init__(self, input_channels, num_channels, kernel_sizes, strides, paddings, use_bn, scaler, input_length = None):

        super(HalcyonInceptionBlock, self).__init__()

        self.input_channels = input_channels
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.use_bn = use_bn
        self.scaler = scaler
        self.input_length = input_length

        if self.input_length is not None:
            self.avg_padding = self._calculate_padding(self.input_length, self.strides[4], self.kernel_sizes[4])
        else:
            self.avg_padding = 0

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
            nn.AvgPool1d(kernel_sizes[4], strides[4], padding = self.avg_padding),
            HalcyonCNNBlock(input_channels, int(num_channels[4] * scaler), 1, strides[4], paddings[4], use_bn[4])
        )
        
    def forward(self, x):

        if self.input_length is None:
            self.input_length = x.shape[2]
            self.cnn5[0].padding = self._calculate_padding(self.input_length, self.strides[4], self.kernel_sizes[4])
        
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x3 = self.cnn3(x)
        x4 = self.cnn4(x)
        x5 = self.cnn5(x)

        x = torch.cat([x1, x2, x3, x4, x5], dim = 1)
        return x

    def _calculate_padding(self, l, s, k):

        return int((l*s - s + k - l)/2)

class HalcyonLSTM(nn.Module):
    """
    Halcyon does not initiate the decoder state with the last hidden state from the encoder (?)
    therefore we already initiate the states with zeros

    https://github.com/relastle/halcyon/blob/ca523b23bdc0778aaea06ef5998f5bd6aa9bf8f8/halcyon/ml/net/decoder/inference_decoder.py#L131-L148
    """


    def __init__(self, input_size, hidden_size, num_layers, bidirectional, proj_size):
        super(HalcyonLSTM, self).__init__()

        # this is to make it compatible with S2SModel
        self.bidirectional = False

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, bidirectional = bidirectional, proj_size = proj_size)

    def forward(self, x):

        output, hidden = self.lstm(x)
        last_hidden = hidden[0][-1, :, :].unsqueeze(0)
        output = output[:, :, output.shape[2]//2:]

        hidden = (
            torch.zeros(last_hidden.shape, device = last_hidden.device, requires_grad = True), 
            torch.zeros(last_hidden.shape, device = last_hidden.device, requires_grad = True)
        )
        return output, hidden
