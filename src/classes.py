import os
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler, DataLoader
from abc import abstractmethod
import numpy as np
import random
from tqdm import tqdm
import warnings
from pathlib import Path

from utils import read_metadata, decode_batch_greedy_ctc
from read import read_fast5
from normalization import normalize_signal_from_read_data
from constants import CTC_BLANK, BASES_CRF, S2S_PAD, S2S_EOS, S2S_SOS, S2S_OUTPUT_CLASSES
from evaluation import alignment_accuracy
from layers.bonito import CTC_CRF

class BaseModel(nn.Module):
    """Abstract class for basecaller models

    It contains some basic methods: train, validate, predict, ctc_decode...
    Since most models follow a similar style.
    """
    
    def __init__(self, device, dataloader_train, dataloader_validation, 
                 optimizer = None, schedulers = dict(), criterions = dict(), clipping_value = 2, use_sam = False):
        super(BaseModel, self).__init__()
        
        self.device = device
        
        # data
        self.dataloader_train = dataloader_train
        self.dataloader_validation = dataloader_validation

        # optimization
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.criterions = criterions
        self.clipping_value = clipping_value
        self.use_sam = use_sam
        
        self.init_weights()
        self.stride = self.get_stride()
        
    @abstractmethod
    def forward(self, batch):
        """Forward through the network
        """
        raise NotImplementedError()
    
    def train_step(self, batch):
        """Train a step with a batch of data
        
        Args:
            batch (dict): dict with keys 'x' (batch, len) 
                                         'y' (batch, len)
        """
        
        self.train()
        x = batch['x'].to(self.device)
        x = x.unsqueeze(1) # add channels dimension
        p = self.forward(x) # forward through the network
        y = batch['y'].to(self.device)
        
        loss, losses = self.calculate_loss(y, p)
        self.optimize(loss)
        
        return losses, p
    
    def validation_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            x = x.unsqueeze(1) # add channels dimension
            p = self.forward(x) # forward through the network
            y = batch['y'].to(self.device)
            
            _, losses = self.calculate_loss(y, p)
            
        return losses, p
    
    def predict_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            x = x.unsqueeze(1)
            p = self.forward(x)
            
        return p
    
    @abstractmethod    
    def decode(self, p, greedy = True):
        """Abstract method that is used to call the decoding approach for
        evaluation metrics during training and evaluation. 
        For example, it can be as simple as argmax for class prediction.
        
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        raise NotImplementedError()

    @abstractmethod    
    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels
            p (tensor): tensor with predictions
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        raise NotImplementedError()
        return loss, losses
    
    
    def optimize(self, loss):
        """Optimizes the model by calculating the loss and doing backpropagation
        
        Args:
            loss (float): calculated loss that can be backpropagated
        """
        
        if self.use_sam: 
            raise NotImplementedError()
            # TODO
            # it is tricky how to use this SAM thing (https://github.com/davda54/sam)
            # because we have to calculate the loss twice, so we have to find a way
            # to make this general
            # also, it is unclear where to put the gradient clipping

            # loss.backward()
            # self.optimizer.first_step(zero_grad=True)
            # loss, losses = self.calculate_loss(y, p)
            # self.optimizer.second_step(zero_grad=True)
        else:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
            self.optimizer.step()
            
        for scheduler in self.schedulers.values():
            if scheduler:
                scheduler.step()
            
        return None

    def evaluate(self, batch, predictions):
        """Evaluate the predictions by calculating the accuracy
        
        Args:
            batch (dict): dict with tensor with [batch, len] in key 'y'
            predictions (list): list of predicted sequences as strings
        """
        y = batch['y'].cpu().numpy()
        y_list = self.dataloader_train.dataset.encoded_array_to_list_strings(y)
        accs = list()
        for i, sample in enumerate(y_list):
            accs.append(alignment_accuracy(sample, predictions[i]))
            
        return {'metric.accuracy': accs}
    
    def init_weights(self):
        """Initialize weights from uniform distribution
        """
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self):
        """Count trainable parameters in model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, checkpoint_file):
        """Save the model state
        """
        save_dict = {'model_state': self.state_dict(), 
                     'optimizer_state': self.optimizer.state_dict()}
        for k, v in self.schedulers.items():
            save_dict[k + '_state'] = v.state_dict()
        torch.save(save_dict, checkpoint_file)
    
    def load(self, checkpoint_file):
        """TODO"""
        raise NotImplementedError()
        
    def get_stride(self):
        """Gives the total stride of the model
        """
        return None
        
    @abstractmethod
    def load_default_configuration(self, default_all = False):
        """Method to load default model configuration
        """
        raise NotImplementedError()

class BaseModelCTC(BaseModel):
    
    def __init__(self, blank = CTC_BLANK, *args, **kwargs):
        """
        Args:   
            blank (int): class index for CTC blank
        """
        super(BaseModelCTC, self).__init__(*args, **kwargs)

        self.criterions['ctc'] = nn.CTCLoss(blank = blank, zero_infinity = True).to(self.device)

    def decode(self, p, greedy = True):
        """Decode the predictions
         
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        if greedy:
            return self.decode_ctc_greedy(p)
        else:
            return self.decode_ctc_beamsearch(p)

    def decode_ctc_greedy(self, p):
        """Predict the bases in a greedy approach
        Args:
            p (tensor): [len, batch, classes]
        """
        p = p.detach()
        p = p.argmax(-1).permute(1, 0)
        p = p.cpu().numpy()
        decoded_predictions = decode_batch_greedy_ctc(y = p, 
                                                      decode_dict = self.dataloader_train.dataset.decoding_dict, 
                                                      blank_label = CTC_BLANK)
        return decoded_predictions

    def decode_ctc_beamsearch(self, p):
        raise NotImplementedError()
            
    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        loss = self.calculate_ctc_loss(y, p)
        losses = {'loss.global': loss.item(), 'loss.ctc': loss.item()}

        return loss, losses

    def calculate_ctc_loss(self, y, p):
        """Calculates the ctc loss
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
        """
        
        y_len = torch.sum(y != CTC_BLANK, axis = 1).to(self.device)
        p_len = torch.full((p.shape[1], ), p.shape[0]).to(self.device)
        
        loss = self.criterions["ctc"](p, y, p_len, y_len)
        
        return loss

class BaseModelCRF(BaseModel):
    
    def __init__(self, state_len = 4, alphabet = BASES_CRF, *args, **kwargs):
        """
        Args:
            state_len (int): k-mer length for the states
            alphabet (str): bases available for states, defaults 'NACGT'
        """
        super(BaseModelCRF, self).__init__(*args, **kwargs)

        self.alphabet = alphabet
        self.state_len = alphabet
        self.seqdist = CTC_CRF(state_len = state_len, alphabet = alphabet)
        self.criterions = {'crf': self.seqdist.ctc_loss}

        
    def decode(self, p, greedy = True):
        """Decode the predictions
        
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        if greedy:
            return self.decode_crf_greedy(p)
        else:
            return self.decode_crf_beamsearch(p)
    
    def decode_crf_greedy(self, y):
        """Predict the sequences using a greedy approach
        
        Args:
            y (tensor): tensor with scores in shape [timesteps, batch, classes]
        Returns:
            A (list) with the decoded strings
        """
        scores = self.seqdist.posteriors(y.to(torch.float32)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        return [self.seqdist.path_to_str(y) for y in tracebacks.cpu().numpy()]

    def decode_crf_beamsearch(self, y):
        raise NotImplementedError()

    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        loss = self.calculate_crf_loss(y, p)
        losses = {'loss.global': loss.item(), 'loss.crf': loss.item()}

        return loss, losses

    def calculate_crf_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """

        y_len = torch.sum(y != CTC_BLANK, axis = 1).to(self.device)
        loss = self.criterions['crf'](scores = p, 
                                      targets = y, 
                                      target_lengths = y_len, 
                                      loss_clip = 10, 
                                      reduction='mean', 
                                      normalise_scores=True)
        return loss

class BaseModelImpl(BaseModelCTC, BaseModelCRF):

    def __init__(self, model_type, *args, **kwargs):
        super(BaseModelImpl, self).__init__(*args, **kwargs)

        valid_model_types = ['ctc', 'crf']
        if model_type not in valid_model_types:
            raise ValueError('Given model_type: ' + str(model_type) + ' is not valid. Valid options are: ' + str(valid_model_types))
        self.model_type = model_type

    def decode(self, p, greedy = True):
        """Decode the predictions
         
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        
        if self.model_type == 'ctc':
            return BaseModelCTC.decode(self, p, greedy)
        if self.model_type == 'crf':
            return BaseModelCRF.decode(self, p, greedy)
        
    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        if self.model_type == 'ctc':
            return BaseModelCTC.calculate_loss(self, y, p)
        if self.model_type == 'crf':
            return BaseModelCRF.calculate_loss(self, y, p)

class BaseModelS2S(BaseModel):
    
    def __init__(
        self, 
        convolution = None, 
        encoder = None, 
        decoder = None, 
        scheduled_sampling = 0, 
        token_sos = S2S_SOS,
        token_eos = S2S_EOS,
        token_pad = S2S_PAD,
        out_classes = S2S_OUTPUT_CLASSES,
        *args, **kwargs):

        """Base model for Seq2Seq

        Args:
            convolution (nn.Module): convolution model that is applied before the encoder
            encoder (nn.Module): rnn for the encoder
            decoder (nn.Module): s2s decoder with embedding, rnn, attention and linear output
            scheduled_sampling (float): probability of using the correct token during training
            token_sos (int): value that indicates start of sentence
            token_eos (int): value that indicates end of sentence
            token_pad (int): value used for padding sentences

        The forward is done using a for loop, it stops if all the samples of the
        batch predict an eos_token. Max length is not defined and it goes at least
        as many times as the amount of timesteps in the output of the encoder.
        """

        super(BaseModelS2S, self).__init__(*args, **kwargs)

        self.convolution = convolution
        self.encoder = encoder
        self.decoder = decoder

        self.token_sos = token_sos
        self.token_eos = token_eos
        self.token_pad = token_pad
        self.out_classes = out_classes

        self.criterions['ce'] = nn.NLLLoss()
        self.scheduled_sampling = scheduled_sampling

    def forward(self, x, y = None, forced_teaching = 0):
        """Forward through the network

        Args:
            x (tensor): tensor with raw signal in shape [batch, channels, len]
            y (tensor): tensor with the encoded labels as integers [batch, len]
            forced_teaching (float): float between 0 and 1 that determines
                the probability of using the true label for a step during decoding

            Returns:
                tensor with shape
        """

        if forced_teaching > 0 and y is None:
            raise ValueError('y must be given if forced_teaching > 0')

        x = self.convolution(x)
        x = x.permute(2, 0, 1)
        enc_out, hidden = self.encoder(x) 

        # get the relevant hidden states
        hidden = self._concat_hiddens(hidden)

        if not isinstance(hidden, tuple):
            hidden = (hidden)

        # keep track of which samples have predicted eos
        ended_sequences = torch.ones(enc_out.shape[1], device = self.device)
        encoder_timesteps = enc_out.shape[0]

        ## [len, batch, classes]
        outputs = torch.zeros(encoder_timesteps, enc_out.shape[1], self.out_classes).to(self.device)
        outputs[1:, :, self.token_pad] = 1 # fill with padding predictions
        outputs[0, :, self.token_sos] = 1 # first token is always SOS

        # first input is the SOS token
        dec_in = torch.full((enc_out.shape[1], ), fill_value = self.token_sos, device = self.device)
        # initial attention is all zeros except first timepoint so that it can look
        # at all the timepoints
        last_attention = torch.zeros((encoder_timesteps, enc_out.shape[1]), device = self.device)
        last_attention[0, :] = 1

        for t in range(1, encoder_timesteps):

            dec_out, hidden, last_attention = self.decoder(dec_in, hidden, enc_out, last_attention)
            outputs[t, :, :] = dec_out

            teacher_force = random.random() < forced_teaching
            #if teacher forcing, use actual next token as next input
            #if not, use predicted tokens                
            if teacher_force:
                dec_in = y[:, t] # [batch]
            else:   
                dec_in = dec_out.argmax(2).squeeze(0) # [batch]
                
            ended_sequences[torch.where(dec_out.argmax(2).squeeze(0) == self.token_eos)[0]] = 0
            if torch.sum(ended_sequences) == 0:
                break    

        return outputs        

    def train_step(self, batch):
        """Train a step with a batch of data
        
        Args:
            batch (dict): dict with keys 'x' (batch, len) 
                                         'y' (batch, len)
        """
        
        self.train()
        x = batch['x'].to(self.device)
        x = x.unsqueeze(1) # add channels dimension
        y = batch['y'].to(self.device)
        y = y.to(int)
        p = self.forward(x, y, self.scheduled_sampling) # forward through the network
        
        loss, losses = self.calculate_loss(y, p.permute(1, 2, 0))
        self.optimize(loss)
        
        return losses, p
    
    def validation_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            x = x.unsqueeze(1) # add channels dimension
            y = batch['y'].to(self.device)
            y = y.to(int)
            p = self.forward(x, None, 0.0) # forward through the network
            
            _, losses = self.calculate_loss(y, p.permute(1, 2, 0))
            
        return losses, p
    
    def predict_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            x = x.unsqueeze(1)
            p = self.forward(x, None, 0.0)
            
        return p

    def calculate_loss(self, y, p):
        """Calculate CE loss

        Args:
            y (tensor): shape [batch, len]
            p (tensor): shape [batch, channels, len]
        """

        loss = self.criterions['ce'](p, y[:, :p.shape[2]])
        losses = {'loss.global': loss.item(), 'loss.ce': loss.item()}

        return loss, losses
     
    def decode(self, p, greedy = False):
        """Decode the predictions into sequences

        Args:
            p (tensor): tensor of predictions with shape [timesteps, batch, channels]
            greedy (bool): whether to use greedy decoding or beam search

        Returns:
            A `list` with the decoded sequences as strings.
        """

        if greedy:
            return self.decode_greedy(p)
        else:
            return self.decode_beamsearch(p)

    def decode_greedy(self, p):
        
        p = p.detach()
        p = p.permute(1, 0, 2) # [batch, len, channels]
        p = p.argmax(-1) # get most probable 
        p = p.cpu().numpy()
        p = p.astype(str)

        # replace tokens with nothing
        for k in [self.token_sos, self.token_eos, self.token_pad]:
            p[p == str(k)] = ''
        
        # replace predictions with bases
        for k, v in self.dataloader_train.dataset.decoding_dict.items():
            p[p == str(k)] = v

        # join everything
        decoded_sequences = ["".join(i) for i in p.tolist()]

        return decoded_sequences

    def decode_beamsearch(self):
        raise NotImplementedError()

    def _concat_hiddens(self, hidden):
        """Concatenates the hidden states output of an RNN

        The output of the RNN in the encoder outputs the last hidden states
        of all the RNN layers. We only want the last (or two last if bidirectional)
        This function extracts the relevant hidden states, and cell states
        for LSTM, to be used as input for the decoder.

        Args:
            hidden (tuple, tensor): with shape [len, batch, hidden]

        Returns:
            Same shaped tensor with last two hidden states concatenated in 
            the hidden dimension
        """

        if isinstance(hidden, tuple):
            return (self._concat_hiddens(hidden[0]), self._concat_hiddens(hidden[1]))

        if self.encoder.bidirectional:
            return torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim = -1).unsqueeze(0)
        else:
            return hidden[-1, :, :].unsqueeze(0)

class BaseNanoporeDataset(Dataset):
    """Base dataset class that contains Nanopore data
    
    The simplest class that handles a hdf5 file that has two datasets
    named 'x' and 'y'. The first one contains an array of floats with
    the raw data normalized. The second one contains an array of 
    byte-strings with the bases appended with ''.
    
    This dataset already takes case of shuffling, for the dataloader set
    shuffling to False.
    
    Args:
        data (str): dir with the npz files
        decoding_dict (dict): dictionary that maps integers to bases
        encoding_dict (dict): dictionary that maps bases to integers
        split (float): fraction of samples for training
        randomizer (bool): whether to randomize the samples order
        seed (int): seed for reproducible randomization
        s2s (bool): whether to encode for s2s models
        token_sos (int): value used for encoding start of sequence
        token_eos (int): value used for encoding end of sequence
        token_pad (int): value used for padding all the sequences (s2s and not s2s)
    """

    def __init__(self, data_dir, decoding_dict, encoding_dict, 
                 split = 0.95, shuffle = True, seed = None,
                 s2s = False, token_sos = S2S_SOS, token_eos = S2S_EOS, token_pad = S2S_PAD):
        super(BaseNanoporeDataset, self).__init__()
        
        self.data_dir = data_dir
        self.decoding_dict = decoding_dict
        self.encoding_dict = encoding_dict
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        
        self.files_list = self._find_files()
        self.num_samples_per_file = self._get_samples_per_file()
        self.total_num_samples = np.sum(np.array(self.num_samples_per_file))
        self.train_files_idxs = set()
        self.validation_files_idxs = set()
        self.train_idxs = list()
        self.validation_idxs = list()
        self.train_sampler = None
        self.validation_sampler = None
        self._split_train_validation()
        self._get_samplers()
        
        self.loaded_train_data = None
        self.loaded_validation_data = None
        self.current_loaded_train_idx = None
        self.current_loaded_validation_idx = None

        self.s2s = s2s
        self.token_sos = token_sos
        self.token_eos = token_eos
        self.token_pad = token_pad

        self._check()
    
    def __len__(self):
        """Number of samples
        """
        return self.total_num_samples
        
    def __getitem__(self, idx):
        """Get a set of samples by idx
        
        If the datafile is not loaded it loads it, otherwise
        it uses the already in memory data.
        
        Returns a dictionary
        """
        if idx[0] in self.train_files_idxs:
            if idx[0] != self.current_loaded_train_idx:
                self.loaded_train_data = self.load_file_into_memory(idx[0])
                self.current_loaded_train_idx = idx[0]
            return self.get_data(data_dict = self.loaded_train_data, idx = idx[1])
        elif idx[0] in self.validation_files_idxs:
            if idx[0] != self.current_loaded_validation_idx:
                self.loaded_validation_data = self.load_file_into_memory(idx[0])
                self.current_loaded_validation_idx = idx[0]
            return self.get_data(data_dict = self.loaded_validation_data, idx = idx[1])
        else:
            raise IndexError('Given index not in train or validation files indices: ' + str(idx[0]))
    
    def _check(self):
        """Check for possible problems
        """

        # check that the encoding dict does not conflict with S2S tokens
        if self.s2s:
            s2s_tokens = (self.token_eos, self.token_sos, self.token_pad)
            for v in self.encoding_dict.values():
                assert v not in s2s_tokens

    def _find_files(self):
        """Finds list of files to read
        """
        l = list()
        for f in os.listdir(self.data_dir):
            if f.endswith('.npz'):
                l.append(f)
        l = sorted(l)
        return l
    
    def _get_samples_per_file(self):
        """Gets the number of samples per file from the file name
        """
        l = list()
        for f in self.files_list:
            metadata = read_metadata(os.path.join(self.data_dir, f))
            l.append(metadata[0][1][0]) # [array_num, shape, first elem shape]
        return l
    
    def _split_train_validation(self):
        """Splits datafiles and idx for train and validation according to split
        """
        
        # split train and validation data based on files
        num_train_files = int(len(self.files_list) * self.split)
        num_validation_files = len(self.files_list) - num_train_files
        
        files_idxs = list(range(len(self.files_list)))
        if self.shuffle:
            if self.seed:
                random.seed(self.seed)
            random.shuffle(files_idxs)
            
        self.train_files_idxs = set(files_idxs[:num_train_files])
        self.validation_files_idxs = set(files_idxs[num_train_files:])
        
        # shuffle indices within each file and make a list of indices (file_idx, sample_idx)
        # as tuples that can be iterated by the sampler
        for idx in self.train_files_idxs:
            sample_idxs = list(range(self.num_samples_per_file[idx]))
            if self.shuffle:
                if self.seed:
                    random.seed(self.seed)
                random.shuffle(sample_idxs)
            for i in sample_idxs:
                self.train_idxs.append((idx, i))
        
        for idx in self.validation_files_idxs:
            sample_idxs = list(range(self.num_samples_per_file[idx]))
            if self.shuffle:
                if self.seed:
                    random.seed(self.seed)
                random.shuffle(sample_idxs)
            for i in sample_idxs:
                self.validation_idxs.append((idx, i))
                
        return None
    
    def _get_samplers(self):
        """Add samplers
        """
        self.train_sampler = IdxSampler(self.train_idxs, data_source = self)
        self.validation_sampler = IdxSampler(self.validation_idxs, data_source = self)
        return None
            
    def load_file_into_memory(self, idx):
        """Loads a file into memory and processes it
        """
        arr = np.load(os.path.join(self.data_dir, self.files_list[idx]))
        x = arr['x']
        y = arr['y']
        return self.process({'x':x, 'y':y})
    
    def get_data(self, data_dict, idx):
        """Slices the data for given indices
        """
        return {'x': data_dict['x'][idx], 'y': data_dict['y'][idx]}
    
    def process(self, data_dict):
        """Processes the data into a ready for training format
        """
        
        y = data_dict['y']
        if y.dtype != 'U1':
            y = y.astype('U1')
        if self.s2s:
            y = self.encode_s2s(y)
        else:
            y = self.encode(y)
        data_dict['y'] = y
        return data_dict
    
    def encode(self, y_arr):
        """Encode the labels
        """
        
        new_y = np.full(y_arr.shape, self.token_pad, dtype=int)
        for k, v in self.encoding_dict.items():
            new_y[y_arr == k] = v
        return new_y

    def encode_s2s(self, y_arr):
    
        new_y = np.full(y_arr.shape, self.token_pad, dtype=int)
        # get the length of each sample to add eos token at the end
        sample_len = np.sum(y_arr != '', axis = 1)
        # array with sos_token to append at the begining
        sos_token = np.full((y_arr.shape[0], 1), self.token_sos, dtype=int)
        # replace strings for integers according to encoding dict
        for k, v in self.encoding_dict.items():
            if v is None:
                continue
            new_y[y_arr == k] = v
        # replace first padding for eos token
        for i, s in enumerate(sample_len):
            new_y[i, s] = self.token_eos
        # add sos token and slice of last padding to keep same shape
        new_y = np.concatenate([sos_token, new_y[:, :-1]], axis = 1)
        return new_y
    
    def encoded_array_to_list_strings(self, y):
        """Convert an encoded array back to a list of strings

        Args:
            y (array): with shape [batch, len]
        """

        y = y.astype(str)
        if self.s2s:
            # replace tokens with nothing
            for k in [self.token_sos, self.token_eos, self.token_pad]:
                y[y == str(k)] = ''
        else:
            y[y == str(self.token_pad)] = ''
        # replace predictions with bases
        for k, v in self.decoding_dict.items():
            y[y == str(k)] = v

        # join everything
        decoded_sequences = ["".join(i) for i in y.tolist()]
        return decoded_sequences




class IdxSampler(Sampler):
    """Sampler class to not sample from all the samples
    from a dataset.
    """
    def __init__(self, idxs, *args, **kwargs):
        super(IdxSampler, self).__init__(*args, **kwargs)
        self.idxs = idxs

    def __iter__(self):
        return iter(self.idxs)

    def __len__(self):
        return len(self.idxs)
        
class BaseFast5Dataset(Dataset):
    """Base dataset class that iterates over fast5 files for basecalling
    
    Args:
        data_dir (str): dir with fast5files
    """

    def __init__(self, data_dir, recursive = True):
        super(BaseFast5Dataset, self).__init__()
    
        self.data_dir = data_dir
        self.recursive = recursive
        self.data_files = self.find_all_fast5_files()

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        read_data = read_fast5(self.data_files[idx])
        return read_data
        
    def find_all_fast5_files(self):
        """Find all fast5 files in a dir recursively
        """
        # find all the files that we have to process
        files_list = list()
        for path in Path(self.data_dir).rglob('*.fast5'):
            files_list.append(str(path))
        return files_list
        
class BaseBasecallDataset(Dataset):
    """A simple dataset for basecalling purposes
    """
    def __init__(self, x):
        self.x = x
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return {'x': self.x[idx, :]}

class BaseBasecaller:
    """A base Basecaller class that is used to basecall complete reads
    """
    
    def __init__(self, model, chunk_size, overlap, batch_size):
        """
        Args:
            model (nn.Module): a model that has the following methods:
                predict, decode
            chunk_size (int): length of the chunks that a read will be divided into
            overlap (int): amount of overlap between consecutive chunks
            batch_size (int): batch size to forward through the network
        """
        
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size
        
    def basecall(self, files_list, output_dir, reads_per_file = 0, greedy = True, silent = False):
        """
        """
        
        read_counter = 0
        file_gen = self.output_file_gen()
        output_file = next(file_gen)
        
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        # iterate read by read
        with tqdm(disable = silent) as pbar:
            for fast5_file in files_list:
                read_data = read_fast5(fast5_file)
                
                for read_id, read_values in read_data.items():
                    
                    signal = self.process_read(read_values)
                    # chunk
                    chunks = self.chunk(signal, self.chunk_size, self.overlap)
                    
                    # make a dataset
                    dataloader = DataLoader(BaseBasecallDataset(chunks), self.batch_size, shuffle = False, drop_last = False)
                    
                    preds = list()
                    for batch in dataloader:
                        # forward through model
                        
                        p = self.model.predict_step(batch)
                        preds.append(p)
                    p = torch.cat(preds, dim = 1)
                    p = p.permute(1, 0, 2)
                    
                    stride = self.model.stride
                    if not stride:
                        stride = int(self.chunk_size // p.shape[1])
                        warnings.warn(('Model has no attribute stride, ideally it would have that, ' + 
                                       'the stride has been deduced based on the output size of the ' + 
                                       'predictions as being: ' + str(stride)))

                    # stich
                    # TODO: this is not compatible with new stich implementations
                    long_p = self.stich(p, self.chunk_size, self.overlap, len(signal), stride) 
                    long_p = long_p.unsqueeze(1)
                    
                    # decode
                    pred_str = self.model.decode(long_p, greedy = greedy)[0]
                    
                    with open(os.path.join(output_dir, output_file), 'a') as f:
                        f.write('>' + str(read_id) + '\n')
                        f.write(pred_str + '\n')
                        
                    # write to file
                    pbar.update(1)
                    read_counter += 1
                    
                    # if we have a limit of reads per file
                    if reads_per_file > 0:
                        # if we have reached the limit
                        if read_counter == reads_per_file :
                            # get the next file name
                            output_file = next(file_gen)
                            # reset the counter
                            read_counter = 0
        return None
    
    def process_read(self, read_data):
        """
        Process the read data as adequate
        
        Args:
            read_data (ReadData): object with all the read data values
            
        Returns:
            A torch tensor with the processed signal
        """
        signal = normalize_signal_from_read_data(read_data)
        signal = torch.from_numpy(signal)
        
        return signal
    
    
    @abstractmethod
    def stich(self, chunks, *args, **kwargs):
        """
        Stitch chunks together with a given overlap
        
        Args:
            chunks (tensor): predictions with shape [samples, length, classes]
        """
        raise NotImplementedError()
        
    def chunk(self, signal, chunksize, overlap):
        """
        Convert a read into overlapping chunks before calling
        
        Since it is unlikely that the length of the signal is perfectly divisible 
        by the chunksize, the first chunk starts from 0, but the second chunk will
        start from the datapoint that makes it to have a perfect ending.
        
        Args:
            signal (tensor): 1D tensor with the raw signal
            chunksize (int): size of each chunk
            overlap (int): datapoints overlap between chunks
            
        Copied from https://github.com/nanoporetech/bonito
        """
        T = signal.shape[0] # 
        if chunksize == 0:
            chunks = signal[None, :]
        elif T < chunksize:
            chunks = torch.nn.functional.pad(signal, (chunksize - T, 0))[None, :]
        else:
            stub = (T - overlap) % (chunksize - overlap)
            chunks = signal[stub:].unfold(0, chunksize, chunksize - overlap)
            if stub > 0:
                chunks = torch.cat([signal[None, :chunksize], chunks], dim=0)
        return chunks


    def stitch_by_stride(self, chunks, chunksize, overlap, length, stride, reverse=False):
        """
        Stitch chunks together with a given overlap
        
        This works by calculating what the overlap should be between two outputed
        chunks from the network based on the stride and overlap of the inital chunks.
        The overlap section is divided in half and the outer parts of the overlap
        are discarded and the chunks are concatenated. There is no alignment.
        
        Chunk1: AAAAAAAAAAAAAABBBBBCCCCC
        Chunk2:               DDDDDEEEEEFFFFFFFFFFFFFF
        Result: AAAAAAAAAAAAAABBBBBEEEEEFFFFFFFFFFFFFF
        
        Args:
            chunks (tensor): predictions with shape [samples, length, classes]
            chunk_size (int): initial size of the chunks
            overlap (int): initial overlap of the chunks
            length (int): original length of the signal
            stride (int): stride of the model
            reverse (bool): if the chunks are in reverse order
            
        Copied from https://github.com/nanoporetech/bonito
        """
        if chunks.shape[0] == 1: return chunks.squeeze(0)

        semi_overlap = overlap // 2
        start, end = semi_overlap // stride, (chunksize - semi_overlap) // stride
        stub = (length - overlap) % (chunksize - overlap)
        first_chunk_end = (stub + semi_overlap) // stride if (stub > 0) else end

        if reverse:
            chunks = list(chunks)
            return torch.cat([
                chunks[-1][:-start], *(x[-end:-start] for x in reversed(chunks[1:-1])), chunks[0][-first_chunk_end:]
            ])
        else:
            return torch.cat([
                chunks[0, :first_chunk_end], *chunks[1:-1, start:end], chunks[-1, start:]
            ])
        
    def output_file_gen(self):
        """An infinite name generator
        """
        i = -1
        while True:
            i += 1
            yield 'basecalls_' + str(i) + '.fasta'