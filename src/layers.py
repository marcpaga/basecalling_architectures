"""Here the different custom layers are defined 
"""

from torch import nn

class BonitoLSTM(nn.Module):
    """Single LSTM RNN layer that can be reversed.
    Useful to stack forward and reverse layers one after the other.
    The default in pytorch is to have the forward and reverse in
    parallel.
    """
    def __init__(self, in_channels, out_channels, reverse = False):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            reverse (bool): whether to the rnn direction is reversed
        """
        super(BonitoLSTM, self).__init__()
        
        self.rnn = nn.LSTM(in_channels, out_channels, num_layers = 1, bidirectional = False, bias = True)
        self.reverse = reverse
        
    def forward(self, x):
        if self.reverse: x = x.flip(0)
        y, h = self.rnn(x)
        if self.reverse: y = y.flip(0)
        return y