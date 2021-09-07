from torch import nn

class BaseModel(nn.Module):
    
    def __init__(self, device, optimizers, schedulers, criterions, clipping_value = 2, use_sam = False):
        super(BaseModel, self).__init__()
        
        self.device = device
        
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.criterions = criterions
        
        self.clipping_value = clipping_value
        self.use_sam = use_sam
        
    def train_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        raise NotImplementedError()
        return predictions, losses
    
    def validate_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        raise NotImplementedError()
        return predictions, losses
        
    def predict(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input only
        """
        raise NotImplementedError()
        return predictions
        
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