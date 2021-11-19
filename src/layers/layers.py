"""Here the different custom layers are defined 
"""

import torch
from torch import nn
import math
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models

    Non-trainable pre-defined position encoding based on sinus and cosinus waves.

    Copied from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class DecoderS2S(nn.Module):

    def __init__(self, embedding, rnn, attention, out_linear, upstream_attention = True):
        """Seq2Seq decoder

        Args:
            embedding (nn.Module): embedding module for the input of the previous token
            rnn (nn.Module): RNN decoder layer
            attention (nn.Module): attention module
            out_linear (nn.Module): linear layer that has output classes as output channels
            upstream_attention (bool): whether the attention is applied before the decoder rnn
        """
        super(DecoderS2S, self).__init__()

        self.embedding = embedding
        self.rnn = rnn
        self.attention = attention
        self.out_linear = out_linear
        self.upstream_attention = upstream_attention

        if self.upstream_attention:
            self.concat = nn.Linear()
        else:
            self.concat = nn.Linear()
        

    def forward(self, inputs, hidden, encoder_outputs, last_attention):
        """
        Args:
            inputs (tensor): [batch, len] with integers to be embedded
            hidden (tensor or tuple): [len (1), batch, channels]
            encoder_outputs (tuple): [len, batch, channels]
            last_attention (tensor): [len, batch]

        Returns:
            (tensor): [len (1), batch, output_channels] with logsoftmax at last dim
        """

        if self.upstream_attention:
            return self.forward_upstream(inputs, hidden, encoder_outputs, last_attention)
        else:
            return self.forward_downstream(inputs, hidden, encoder_outputs, last_attention)

    def forward_upstream(self, inputs, hidden, encoder_outputs, last_attention):
        """See `forward` for args
        """
        
        embedded = self.embedding(inputs)
        attn_weights = self.attention(hidden, encoder_outputs, last_attention)

        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        concat_input = torch.cat((embedded, context), 1)
        rnn_input = torch.tanh(self.concat(concat_input))
        rnn_output, hidden = self.rnn(rnn_input, hidden)

        output = F.log_softmax(self.out_linear(rnn_output), dim = -1)

        return output

    def forward_downstream(self, inputs, hidden, encoder_outputs, last_attention):
        """See `forward` for args
        """
        
        embedded = self.embedding(inputs)
        rnn_output, hidden = self.rnn(embedded, hidden)
        
        attn_weights = self.attention(rnn_output, encoder_outputs, last_attention)

        # multiply attention weights to encoder outputs
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = F.log_softmax(self.out_linear(concat_output), dim = -1)

        return output