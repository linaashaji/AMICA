"""
Created on Feb 7

@author: lachaji

"""

import torch.nn as nn
from .agentwise_pooling import AgentWisePooling
from .agentwise_pooling import FullAgentWisePooling

class ZNet(nn.Module):
    """
        Agent-wise pooling + feed-forward
    """
    def __init__(self, feed_forward):
        super().__init__()
        self.agent_wise_pooling = AgentWisePooling()
        self.feed_forward = feed_forward
        
    def forward(self, memory, mask=None):
        """
        memory.shape = (batch_size, obs_length, num_tracks, d_model)
        mask : source mask
             shape = (batch_size, obs_length * num_tracks or 1, obs_length * num_tracks)
        """
        
        Z = self.agent_wise_pooling(memory, mask) 
        # shape = (batch_size, num_tracks, d_model)
        return self.feed_forward(Z)
    
    
class ZNetOne(nn.Module):
    """
        Agent-wise pooling + feed-forward
    """
    def __init__(self, feed_forward):
        super().__init__()
        self.agent_wise_pooling = FullAgentWisePooling()
        self.feed_forward = feed_forward
        
    def forward(self, memory, mask=None):
        """
        memory.shape = (batch_size, obs_length, d_model)
        mask : source mask
             shape = (batch_size, obs_length or 1, obs_length)
        """
        
        Z = self.agent_wise_pooling(memory, mask) 
        return self.feed_forward(Z)