"""
Created on Feb 7

@author: lachaji

"""

import itertools
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """
    Stack of Linear->relu->dropout
    """
    def __init__(self, dims, dropout=0.0, activation = nn.ReLU()):
        """
        dims : list of dimensions of linear projections
        dropout : dropout rate
        """
        super().__init__()
        self.feed_forward = nn.Sequential( *list(itertools.chain(*[
            [ nn.Linear(dims[i], dims[i+1]), activation, nn.Dropout(dropout)] for i in range(len(dims)-1) 
        ]) )[:-2] ) # [:-1] to not have relu and dropout after the last layer
                                 
    def forward(self, x):
        return self.feed_forward(x)