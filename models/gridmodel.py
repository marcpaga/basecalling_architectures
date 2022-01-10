import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from classes import BaseModelImpl
from bonito.model import BonitoModel
from causalcall.model import CausalCallModel
from halcyon.model import HalcyonModel
from mincall.model import MinCallModel
from urnano.model import URNanoModel
from sacall.model import SACallModel
from catcaller.model import CATCallerModel

from torch import nn

class GridAnalysisModel(
    BonitoModel, 
    CausalCallModel, 
    HalcyonModel, 
    MinCallModel,
    SACallModel,
    URNanoModel,
    CATCallerModel,
    BaseModelImpl):

    def __init__(self, cnn_type, encoder_type, decoder_type, use_connector = False, *args, **kwargs):
        super(GridAnalysisModel, self).__init__(decoder_type = decoder_type, *args, **kwargs)

        self.cnn_type = cnn_type
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.use_connector = use_connector

        self.convolution = self.build_cnn()
        self.encoder = self.build_encoder()
        if use_connector:
            self.connector = self.build_connector()
        self.decoder = self.build_decoder()

    def forward(self, x):
        
        # [batch, channels, len]
        x = self.convolution(x)

        # [batch, channels, len]
        if self.use_connector:
            x = x.permute(0, 2, 1)
            # [batch, len, channels]
            x = self.connector(x)
            x = x.permute(0, 2, 1)
            # [batch, channels, len]

        x = x.permute(2, 0, 1) # [len, batch, channels]
        x = self.encoder(x)

        # get rid of RNN hidden states
        if isinstance(x, tuple):
            x = x[0]
        x = self.decoder(x)
        return x

    def build_cnn(self):

        if self.cnn_type == 'bonito':
            defaults = BonitoModel.get_defaults(self)
            cnn = BonitoModel.build_cnn(self)
        elif self.cnn_type == 'catcaller':
            defaults = CATCallerModel.get_defaults(self)
            cnn = CATCallerModel.build_cnn(self)
        elif self.cnn_type == 'causalcall':
            defaults = CausalCallModel.get_defaults(self)
            cnn = CausalCallModel.build_cnn(self)
        elif self.cnn_type == 'halcyon':
            defaults = HalcyonModel.get_defaults(self)
            cnn = HalcyonModel.build_cnn(self, mod = False)
        elif self.cnn_type == 'halcyonmod':
            defaults = HalcyonModel.get_defaults(self)
            cnn = HalcyonModel.build_cnn(self, mod = True)
        elif self.cnn_type == 'mincall':
            defaults = MinCallModel.get_defaults(self)
            cnn = MinCallModel.build_cnn(self)
        elif self.cnn_type == 'sacall':
            defaults = SACallModel.get_defaults(self)
            cnn = SACallModel.build_cnn(self)
        elif self.cnn_type == 'urnano':
            defaults = URNanoModel.get_defaults(self)
            cnn = URNanoModel.build_cnn(self)
        else:
            raise ValueError('invalid cnn_type')

        self.cnn_output_size = defaults['cnn_output_size']
        self.cnn_output_activation = defaults['cnn_output_activation']
        return cnn

    def build_encoder(self):

        if self.encoder_type == 'bonitofwd':
            defaults = BonitoModel.get_defaults(self)
            if self.use_connector:
                input_size = defaults['encoder_input_size']
            else:
                input_size = self.cnn_output_size
            encoder = BonitoModel.build_encoder(self, input_size = input_size, reverse = True)

        elif self.encoder_type == 'bonitorev':
            defaults = BonitoModel.get_defaults(self)
            if self.use_connector:
                input_size = defaults['encoder_input_size']
            else:
                input_size = self.cnn_output_size
            encoder = BonitoModel.build_encoder(self, input_size = input_size, reverse = False)

        elif self.encoder_type == 'catcaller':
            defaults = CATCallerModel.get_defaults(self)
            encoder = CATCallerModel.build_encoder(self)
        
        elif self.encoder_type == 'sacall':
            defaults = SACallModel.get_defaults(self)
            encoder = SACallModel.build_encoder(self)
        
        elif self.encoder_type == 'urnano':
            defaults = URNanoModel.get_defaults(self)
            if self.use_connector:
                input_size = defaults['encoder_input_size']
            else:
                input_size = self.cnn_output_size
            encoder = URNanoModel.build_encoder(self, input_size = input_size)
        
        elif self.encoder_type in ('lstm1', 'lstm3', 'lstm5'):
            defaults = {'encoder_input_size': 256, 'encoder_output_size': 512}
            num_layers = int(list(self.encoder_type)[-1])
            if self.use_connector:
                input_size = defaults['encoder_input_size']
            else:
                input_size = self.cnn_output_size

            encoder =  nn.LSTM(input_size = input_size, hidden_size = 256, num_layers = num_layers, bidirectional = True)
        
        else:
            raise ValueError('invalid rnn_type')

        self.encoder_input_size = defaults['encoder_input_size']
        self.encoder_output_size = defaults['encoder_output_size']
        return encoder

    def build_connector(self):
        if self.cnn_output_activation == 'relu':
            return nn.Sequential(nn.Linear(self.cnn_output_size, self.encoder_input_size), nn.ReLU())
        elif self.cnn_output_activation == 'silu':
            return nn.Sequential(nn.Linear(self.cnn_output_size, self.encoder_input_size), nn.SiLU())
        elif self.cnn_output_activation is None:
            return nn.Sequential(nn.Linear(self.cnn_output_size, self.encoder_input_size))

    def build_decoder(self):
        return BaseModelImpl.build_decoder(self, encoder_output_size = self.encoder_output_size, decoder_type = self.decoder_type)