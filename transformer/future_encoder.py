"""
Created on Feb 7

@author: lachaji

"""

import torch.nn as nn
from .znet import ZNet

class FutureEncoder(nn.Module):
    """
    input embedding + cross-agentformer tranformer block + Position-wise feed forward
    See agentFormer paper : https://arxiv.org/pdf/2103.14023
    """
    def __init__(self, decoder, feed_forward_z):
        super().__init__()
        self.decoder = decoder
        self.posterior_net_generator = ZNet( feed_forward_z )
        
    def forward(self, memory, x, src_mask=None, tgt_mask=None,look_ahead=None):
        """
        x.shape = (batch_size, obs_length, num_tracks, d_model)
        mask : mask
             shape = (batch_size, obs_length * num_tracks or 1, obs_length * num_tracks)
        """
        return self.posterior_net_generator( self.decoder(memory, x, src_mask, tgt_mask, look_ahead), mask=tgt_mask )