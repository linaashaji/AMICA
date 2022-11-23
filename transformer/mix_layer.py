"""
Created on Feb 7

@author: lachaji

"""

import torch.nn as nn

  
class MixLayerLN(nn.Module):
    """
    Multi-head self-attn + position-wise feed forward (AxialMultiHeadAttentionLN)
    """
    
    def __init__(self, size, attn, feed_forward, dropout, pre_ln = False):
        super().__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.size = size
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, seed_length, src_mask=None, src_target_mask=None, mix_mask=None):
        x = self.attn(x, x, x, seed_length, src_mask, src_target_mask, mix_mask)
        
        return self.norm(x + self.dropout(self.feed_forward(x)))