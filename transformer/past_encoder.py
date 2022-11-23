"""
Created on Feb 7

@author: lachaji

"""

import torch.nn as nn
from .znet import ZNet

class PastEncoder(nn.Module):
    """
    input embedding + cross-agentformer tranformer block + Position-wise feed forward
    See agentFormer paper : https://arxiv.org/pdf/2103.14023
    """
    def __init__(self, encoder, feed_forward_z):
        super().__init__()
        self.encoder = encoder
        self.posterior_net_generator = ZNet( feed_forward_z )
        
    def forward(self, memory, tgt_mask=None):
        """
        x.shape = (batch_size, obs_length, num_tracks, d_model)
        mask : mask
             shape = (batch_size, obs_length * num_tracks or 1, obs_length * num_tracks)
        """
        return self.posterior_net_generator( self.encoder(memory, tgt_mask), mask=tgt_mask )