from torch import nn
from torch.utils.data import Dataset
from abc import abstractmethod

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

class BaseDataloader(Dataset):

    def __init__(self, decoding_dict, encoding_dict):
        super(BaseDataloader, self).__init__()

        self.decoding_dict = decoding_dict
        self.encoding_dict = encoding_dict

    @abstractmethod
    def process(self):
        """Abstract method that processes the data into a ready
        for training format
        """
        raise NotImplementedError()

    @abstractmethod
    def encode(self):
       """Abstract method that encodes the labels
       """
       raise NotImplementedError()

    @abstractmethod
    def decode(self):
       """Abstract method that decodes the labels
       """

