import os
from torch import nn
from torch.utils.data import Dataset, Sampler
from abc import abstractmethod
import numpy as np
import random

class BaseModel(nn.Module):
    """ Abstract class for basecaller models
    """
    
    def __init__(self, device, dataloader, 
                 optimizers, schedulers, criterions, clipping_value = 2, use_sam = False):
        super(BaseModel, self).__init__()
        
        self.device = device
        
        # data
        self.dataloader = dataloader

        # optimization
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.criterions = criterions
        self.clipping_value = clipping_value
        self.use_sam = use_sam
        
        self.to(self.device())
        self.init_weights()
    
    @abstractmethod    
    def train_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        raise NotImplementedError()
        return losses, predictions
    
    @abstractmethod
    def validate_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        raise NotImplementedError()
        return losses, predictions
    
    @abstractmethod    
    def predict(self):
        """Abstract method that takes care of the whole prediction and 
        assembly of a set of reads.
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
            loss.backward()
            self.optimizer.first_step(zero_grad=True)
            loss, losses = self.calculate_loss(y, p)
            self.optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
            optimizer.step()
            
        return None
    
    def init_weights(self):
        """Initialize weights from uniform distribution
        """
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self):
        """Count trainable parameters in model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



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
    """

    def __init__(self, data_dir, decoding_dict, encoding_dict, 
                 split = 0.95, shuffle = True, seed = None):
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
            l.append(int(f.split('_')[-1].split('.')[0]))
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
        y = self.encode(y)
        data_dict['y'] = y
        return data_dict
    
    
    def encode(self, y_arr):
        """Encode the labels
        """
        
        new_y = np.zeros(y_arr.shape, dtype=int)
        for k, v in self.encoding_dict.items():
            new_y[y_arr == k] = v
        return new_y
    
    
    def decode(self):
        """Decode the labels
        """
        
        y_arr = y_arr.astype(str)
        for k, v in self.decoding_dict.items():
            y_arr[y_arr == k] = v
        return y_arr
        
        
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