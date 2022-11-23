"""
Created on Feb 7

@author: lachaji

"""

import torch.nn as nn

from. utils import clones
from .sublayer_connection import SubLayerConnection


class DecoderLayer(nn.Module):
    """
    self-attn + src-attn(cross-attn) + feed forward
    """
    
    def __init__(self, size, self_attn=None, src_attn=None, feed_forward=None, dropout=None, pre_ln=False):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        n_sub_connections = 2 if self_attn is None else 3
        self.sublayer = clones( SubLayerConnection(size, dropout, pre_ln), n_sub_connections )
    
    def forward(self, memory, x, src_mask, tgt_mask=None, look_ahead=None):
        m = memory
        i = 0
        if self.self_attn is not None:
            x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, tgt_mask, look_ahead) )
            i+=1
        x = self.sublayer[i](x, lambda x : self.src_attn(x, m, m, src_mask) )
        return self.sublayer[i+1](x, lambda x : self.feed_forward(x) )
    
    
    
class DecoderLayerLN(nn.Module):
    """
    Multi-head self-attn + src-attn(cross-attn) + position-wise feed forward 
    (AxialMultiHeadAttentionLN, TemporalMultiHeadAttentionLN)
    """
    
    def __init__(self, size, self_attn, src_attn=None, feed_forward=None, dropout=None, pre_ln=False):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.size = size
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, memory, x, src_mask, tgt_mask=None, look_ahead=None):
        
        m = memory
        x = self.self_attn(x, x, x, tgt_mask, look_ahead)
        x = self.src_attn(x, m, m, src_mask)
        
        return self.norm(x + self.dropout(self.feed_forward(x)))
    
    
    
class TriDecoderLayer(nn.Module):
    """
    self-attn + src-attn(cross-attn) + feed forward
    """
    
    def __init__(self, size, self_attn=None, src_attn=None, src_attn2=None, feed_forward=None, dropout=None, pre_ln=False):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.src_attn2 = src_attn
        self.feed_forward = feed_forward
        n_sub_connections = 2 if self_attn is None else 3 if src_attn2 is None else 4
        self.sublayer = clones( SubLayerConnection(size, dropout, pre_ln), n_sub_connections )
    
    def forward(self, memory_2, memory_1, x, memory_2_mask=None, memory_1_mask=None, x_mask=None):
        
        '''
            x: input to the decoder
            x_mask: mask applied on the input to the decoder --> probably tgt_mask
            
            memory_1: first memory from bottom to top
            memory_2: second memory from bottom to top
        
        '''

        i = 0
        if self.self_attn is not None:
            x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, x_mask))
            i+=1
        x = self.sublayer[i](x, lambda x : self.src_attn(x, memory_1, memory_1, memory_1_mask))
        x = self.sublayer[i+1](x, lambda x : self.src_attn2(x, memory_2, memory_2, memory_2_mask) )
        return self.sublayer[i+2](x, lambda x : self.feed_forward(x))
    
    
