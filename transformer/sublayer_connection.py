"""
Created on Feb 7

@author: lachaji

"""

import torch.nn as nn
import torch.nn.functional as F

class SubLayerConnection(nn.Module):
    """
    residual connection followed by a layer norm
    """
    def __init__(self, size, dropout=0.0, pre_ln = False):
        """
        pre_ln : wether to do pre-layer normalization (pre_ln) or post layer norm
        """
        super(SubLayerConnection, self).__init__()
        self.pre_ln = pre_ln
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        if self.pre_ln:
            return x + self.dropout( sublayer(self.norm(x)) )
        else:
            return self.norm( x + self.dropout( sublayer(x) ) )