import torch.nn as nn

class MLP(nn.Module):
    """
    This class implements an MLP in Pytorch.
    It handles the different layers and parameters of the model.
    """
    
    def __init__(self, n_input, n_hidden, n_classes):
        """
        
        Args:
          n_input: number of input dimensions
          n_hidden: number of hidden dimensions
          n_classes: number of output dimensions
          
        """        

        super().__init__()
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        
        self.mlp = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_classes)
        )
    
    def forward(self, x):
        """
        Performs forward pass of the input.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        """

        out = self.mlp(x)
        
        return out
