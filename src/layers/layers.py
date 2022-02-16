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


class RNNDecoderS2S(nn.Module):

    def __init__(self, embedding, rnn, out_linear, attention = None, attention_pos = 'upstream'):
        """Seq2Seq decoder

        The rnn should have as many hidden dimensions as the output dimensions of
        the encoder rnn, otherwise you cannot pass the hidden states.

        Upstream attention: attention is applied between the encoder outputs and
            the last hidden state. Then this attention is concatenated to the 
            embeddings and fed as input for the decoder RNN.

        Downstream attention: attention is applied between the output of the 
            decoder and the encoder outputs. Then the output of the decoder
            is concatenated with the attention and fed to the output linear layer.

        No attention: the embedded input is directly fed into the decoder RNN
            and the same for the hidden state tensor.

        Args:
            embedding (nn.Module): embedding module for the input of the previous token
            rnn (nn.Module): RNN decoder layer
            attention (nn.Module): attention module
            out_linear (nn.Module): linear layer that has output classes as output channels
            encoder_hidden (int): number of hidden dimensions of the encoder, consider hidden*2 if bidirectional. If
                None it will be taken from the first forward through the network
            attention_pos (str): "upstream" for attention before the decoder, or "downstream" for attention after decoder
        """
        super(RNNDecoderS2S, self).__init__()

        self.embedding = embedding
        self.rnn = rnn
        self.attention = attention
        self.out_linear = out_linear
        self.attention_pos = attention_pos

        if self.attention_pos == 'upstream':
            self.concat = nn.LazyLinear(self.rnn.input_size)
        elif self.attention_pos == 'downstream':
            self.concat = nn.Linear(self.rnn.hidden_size * 2, self.out_linear.in_features)
        else:
            raise ValueError('attention pos has to be "upstream" or "downstream", given: ' + str(self.attention_pos))
        if self.rnn.bidirectional:
            raise AttributeError('RNN cannot be bidirectional')
        

    def forward(self, inputs, hidden, encoder_outputs, last_attention = None):
        """
        Args:
            inputs (tensor): [batch, len (1)] with integers to be embedded
            hidden (tuple): [len (1), batch, channels]
            encoder_outputs (tuple): [len, batch, channels]
            last_attention (tensor): [len, batch]

        hidden should always be a tuple and it can come from LSTM or GRU

        Returns:
            (tensor): [len (1), batch, output_channels] with logsoftmax at last dim
            hidden (tuple, tensor): [len (1), batch, hidden]
            attention: [len, batch]
        """

        # for multilayer decoder we have to append 0 for the rest of the layers
        if hidden[0].shape[0] < self.rnn.num_layers:
            if len(hidden) == 1:
                hidden = (torch.cat([hidden[0], torch.zeros((self.rnn.num_layers - 1, hidden[0].shape[1], hidden[0].shape[2]), device = encoder_outputs.device)], dim = 0))
            else:
                hidden = (torch.cat([hidden[0], torch.zeros((self.rnn.num_layers - 1, hidden[0].shape[1], hidden[0].shape[2]), device = encoder_outputs.device)], dim = 0), 
                          torch.cat([hidden[1], torch.zeros((self.rnn.num_layers - 1, hidden[1].shape[1], hidden[1].shape[2]), device = encoder_outputs.device)], dim = 0))

        if self.attention is None:
            return self.forward_no_attention(inputs, hidden)
        else:
            if self.attention_pos == 'upstream':
                return self.forward_upstream(inputs, hidden, encoder_outputs, last_attention)
            elif self.attention_pos == 'downstream':
                return self.forward_downstream(inputs, hidden, encoder_outputs, last_attention)

    def forward_no_attention(self, inputs, hidden):
        """Forward without attention
        
        See `forward` for args
        """

        embedded = self.embedding(inputs)
        embedded = embedded.unsqueeze(2).permute(2, 0, 1) # [len, batch, hidden]

        if len(hidden) > 1:
            rnn_output, hidden = self.rnn(embedded, hidden)
        else:
            rnn_output, hidden = self.rnn(embedded, hidden[0])

        output = F.log_softmax(self.out_linear(rnn_output), dim = -1)

        return output, hidden, None

    def forward_upstream(self, inputs, hidden, encoder_outputs, last_attention):
        """Forward with upstream attention
        
        See `forward` for args
        """
        
        embedded = self.embedding(inputs)
        embedded = embedded.unsqueeze(2) # [batch, hidden, len]

        # do hidden[0][0].unsqueeze(0) in case it is a multilayer hidden, so we only
        # want the first one
        # this works for single layer or multilayer
        attn_weights = self.attention(hidden[0][0].unsqueeze(0), encoder_outputs, last_attention) # [len, batch]
        attn_weights = attn_weights.permute(1, 0).unsqueeze(2) # [batch, len, 1]
        encoder_outputs = encoder_outputs.permute(1, 2, 0) # [batch, hidden, len]

        context = torch.bmm(encoder_outputs, attn_weights)
        concat_input = torch.cat((embedded, context), dim = 1).squeeze(2)
        rnn_input = torch.tanh(self.concat(concat_input))

        rnn_input = rnn_input.unsqueeze(0) # [len, batch, hidden]

        if len(hidden) > 1:
            rnn_output, hidden = self.rnn(rnn_input, hidden)
        else:
            rnn_output, hidden = self.rnn(rnn_input, hidden[0])

        output = F.log_softmax(self.out_linear(rnn_output), dim = -1)

        return output, hidden, attn_weights.squeeze(2).permute(1, 0)

    def forward_downstream(self, inputs, hidden, encoder_outputs, last_attention):
        """Forward with downstream attention
        
        See `forward` for args
        """
        
        embedded = self.embedding(inputs)
        embedded = embedded.unsqueeze(2).permute(2, 0, 1) # [len, batch, hidden]
        
        if len(hidden) > 1:
            rnn_output, hidden = self.rnn(embedded, hidden)
        else:
            rnn_output, hidden = self.rnn(embedded, hidden[0])
        
        attn_weights = self.attention(rnn_output, encoder_outputs, last_attention)
        attn_weights = attn_weights.permute(1, 0).unsqueeze(2) # [batch, len, 1]
        encoder_outputs = encoder_outputs.permute(1, 2, 0) # [batch, hidden, len]
        
        context = torch.bmm(encoder_outputs, attn_weights)
        context = context.permute(2, 0, 1)

        concat_input = torch.cat((rnn_output, context), dim = 2)
        concat_output = torch.tanh(self.concat(concat_input))
        
        output = F.log_softmax(self.out_linear(concat_output), dim = -1)

        return output, hidden, attn_weights.squeeze(2).permute(1, 0)