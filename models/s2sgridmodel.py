from seq2seq.model import Seq2Seq

import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from layers.halcyon import HalcyonLSTM

class S2SGridAnalysisModel(Seq2Seq):

    def __init__(self, 
        cnn_type,
        encoder_type,
        decoder_type,
        attention_type, 
        attention_pos,
        monotonic,
        *args, **kwargs):
        super(S2SGridAnalysisModel, self).__init__(*args, **kwargs)

        self.cnn_type = cnn_type
        self.encoder_struct_type = encoder_type
        self.decoder_struct_type = decoder_type
        self.attention_type = attention_type
        self.attention_pos = attention_pos
        self.monotonic = monotonic

        if self.encoder_struct_type == 'shallow':
            encoder_num_layers = 2
        elif self.encoder_struct_type == 'deep':
            encoder_num_layers = 4 
        else:
            raise ValueError('encoder_type has to be "shallow" or "deep", given: ' + str(self.encoder_struct_type))

        if self.decoder_struct_type == 'shallow':
            decoder_num_layers = 1
        elif self.decoder_struct_type == 'deep':
            decoder_num_layers = 3 
        else:
            raise ValueError('decoder_type has to be "shallow" or "deep", given: ' + str(self.decoder_struct_type))

        self.convolution = Seq2Seq.build_cnn(self, cnn_type = cnn_type)
        self.encoder = Seq2Seq.build_encoder(
            self,
            input_size = self.cnn_output_size, 
            output_size = 256, 
            num_layers = encoder_num_layers, 
            bidirectional = True,
        )
        self.encoder = HalcyonLSTM(
            input_size = 243, 
            hidden_size = 128, 
            num_layers = 5, 
            bidirectional = True,
            proj_size = 96,
        )
        self.decoder = Seq2Seq.build_decoder(
            self,
            embedding_dim = 16, 
            rnn_size = 512, 
            rnn_num_layers = decoder_num_layers, 
            attention_type = attention_type, 
            attention_pos = attention_pos,
            monotonic = monotonic
        )
        self.decoder = Seq2Seq.build_decoder(
            self,
            embedding_dim = 16, 
            rnn_size = 96, 
            rnn_num_layers = 5, 
            attention_type = 'dot', 
            attention_pos = 'downstream',
            monotonic = True
        )