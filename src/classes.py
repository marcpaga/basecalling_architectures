from torch import nn
from torch.utils.data import Dataset, Sampler
from abc import abstractmethod
import h5py
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
        
        self.init_weights()
    
    @abstractmethod    
    def train_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        raise NotImplementedError()
        return predictions, losses
    
    @abstractmethod
    def validate_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        raise NotImplementedError()
        return predictions, losses
    
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
    
    @abstractmethod    
    def optimize(self, y, p):
        """Optimizes the model by calculating the loss and doing backpropagation
        
        Args:
            y (tensor): tensor with labels
            p (tensor): tensor with predictions
        Returns:
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        loss, losses = self.calculate_loss(y, p)
        
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
            
        return losses
    
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
    
    Args:
        hdf5_file (str): hdf5 file that has the data
        decoding_dict (dict): dictionary that maps integers to bases
        encoding_dict (dict): dictionary that maps bases to integers
        split (float): fraction of samples for training
        randomizer (bool): whether to randomize the samples order
        seed (int): seed for reproducible randomization
        load_all (bool): whether to load all the data into memory
    """

    def __init__(self, hdf5_file, decoding_dict, encoding_dict, 
                 split = 0.95, shuffle = True, seed = None, load_all = False):
        super(BaseNanoporeDataset, self).__init__()
        
        self.hdf5_file = hdf5_file
        self.decoding_dict = decoding_dict
        self.encoding_dict = encoding_dict
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.load_all = load_all
        
        self.train_idxs = list()
        self.validation_idxs = list()
        self.train_sampler = None
        self.validation_sampler = None
        self.split_train_validation()
        
        
        if self.load_all:
            self.data = self.load_data_into_memory()
        else:
            self.data = dict()

    def __len__(self):
        """Number of samples
        """
        with h5py.File(self.hdf5_file, "r") as datafile:
            return datafile['x'].shape[0]
        
    def __getitem__(self, idx):
        """Get a set of samples by idx
        
        Returns a dictionary
        """
        if self.load_all:
            return self.getitem_memory(idx)
        else:
            return self.getitem_disk(idx)
        
    def load_data_into_memory(self):
        """Loads all the data into memory
        """
        with h5py.File(self.hdf5_file, "r") as datafile:
            x = datafile['x'][:]
            y = datafile['y'][:]
            y = y.astype('U1')
        return self.process({'x':x, 'y':y})
        
    def getitem_disk(self, idx):
        """Loads a portion of the data from disk
        """
        with h5py.File(self.hdf5_file, "r") as datafile:
            x = datafile['x']
            y = datafile['y']
            return {'x':x[idx], 'idx':idx}
            #return self.process({'x':x[idx], 'y':y[idx]})
        
    def getitem_memory(self, idx):
        """Loads a portion of the data from memory
        """
        return {'x':self.data['x'][idx], 'y':self.data['y'][idx]}
        
    def process(self, data_dict):
        """Processes the data into a ready for training format
        """
        
        y = data_dict['y']
        y = y.astype('U1')
        y = self.encode(y)
        data_dict['y'] = y
        return data_dict
    
    def encode(self, y_arr):
        """Encode the labels
        """
        
        for k, v in self.encoding_dict.items():
            y_arr[y_arr == k] = v
        y_arr = y_arr.astype(int)
        return y_arr
    
    def decode(self):
        """Decode the labels
        """
        
        y_arr = y_arr.astype(str)
        for k, v in self.decoding_dict.items():
            y_arr[y_arr == k] = v
        return y_arr
    
    def split_train_validation(self):
        """Splits train and validation idxs and generates samplers
        """
        
        num_samples = self.__len__()
        num_train_samples = int(num_samples * self.split)
        num_validation_samples = num_samples - num_train_samples
        
        samples_idxs = list(range(num_samples))
        if self.shuffle:
            if self.seed:
                random.seed(self.seed)
            random.shuffle(samples_idxs)
        
        self.train_idxs = samples_idxs[:num_train_samples]
        self.validation_idxs = samples_idxs[num_train_samples:]
        self.train_sampler = IdxSampler(idxs = self.train_idxs, data_source = self)
        self.validation_sampler = IdxSampler(idxs = self.validation_idxs, data_source = self)
        
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