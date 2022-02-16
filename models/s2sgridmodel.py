from seq2seq.model import Seq2Seq

class S2SGridAnalysisModel(Seq2Seq):

    def __init__(self, 
        encoder_type,
        decoder_type,
        attention_type, 
        attention_pos,
        monotonic,
        *args, **kwargs):
        super(S2SGridAnalysisModel, self).__init__(*args, **kwargs)


        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.attention_type = attention_type
        self.attention_pos = attention_pos
        self.monotonic = monotonic

        if self.encoder_type == 'shallow':
            encoder_num_layers = 2
        elif self.encoder_type == 'deep':
            encoder_num_layers = 4 
        else:
            raise ValueError('encoder_type has to be "shallow" or "deep", given: ' + str(encoder_type))

        if self.decoder_type == 'shallow':
            decoder_num_layers = 1
        elif self.decoder_type == 'deep':
            decoder_num_layers = 3 
        else:
            raise ValueError('decoder_type has to be "shallow" or "deep", given: ' + str(decoder_type))

        self.convolution = Seq2Seq.build_cnn(self)
        self.encoder = Seq2Seq.build_encoder(
            self,
            input_size = 384, 
            output_size = 256, 
            num_layers = encoder_num_layers, 
            bidirectional = True,
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