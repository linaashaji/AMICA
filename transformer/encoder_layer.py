"""
Created on Feb 7

@author: lachaji

"""

import torch.nn as nn
import torch.nn.functional as F

from. utils import clones
from .sublayer_connection import SubLayerConnection

class EncoderLayer(nn.Module):
    """
    Multi-head self-attn + position-wise feed forward
    """
    
    def __init__(self, size, self_attn, feed_forward, dropout, pre_ln = False):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones( SubLayerConnection( size, dropout, pre_ln), 2 )
        self.size = size
    
    def forward(self, x, mask, look_ahead=None):
        x = self.sublayer[0]( x, lambda x : self.self_attn(x, x, x, mask, look_ahead) )
        return self.sublayer[1](x, self.feed_forward)
    
    
    
class EncoderLayerLN(nn.Module):
    """
    Multi-head self-attn + position-wise feed forward (AxialMultiHeadAttentionLN)
    """
    
    def __init__(self, size, self_attn, feed_forward, dropout, pre_ln = False):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask, look_ahead=None):
        x = self.self_attn(x, x, x, mask, look_ahead)
        return self.norm(x + self.dropout(self.feed_forward(x)))