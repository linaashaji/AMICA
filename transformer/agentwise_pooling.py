"""
Created on Feb 7

@author: lachaji

"""

import torch
import torch.nn as nn

from .mask import get_temporal_mask

class AgentWisePooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, memory, mask=None):
        """
        memory.shape = (batch_size, length, num_tracks, d_model)
        mask : source mask
             shape = (batch_size, length * num_tracks or 1, length * num_tracks)
        """
        #return torch.mean(memory, dim=1)
        if mask is None:
            return torch.mean(memory, dim=1)
        
        batch_size = memory.size(0)
        obs_length = memory.size(1)
        num_tracks = memory.size(2)
        
        if mask.size(-2) == 1:
            pooling_mask = mask.view(batch_size, obs_length, num_tracks)
        else:
            temporal_mask = get_temporal_mask(mask, obs_length, num_tracks) # shape = (batch_size, num_tracks, length, length)
            pooling_mask = ~(torch.sum( temporal_mask, dim=-2 ) == 0) # shape=(batch_size, num_tracks, length)
            pooling_mask = pooling_mask.transpose(1,2) # shape=(batch_size, length, num_tracks)
        
        #print(pooling_mask)
        pooling_mask = pooling_mask.unsqueeze(-1)
        
        result = memory.masked_fill( pooling_mask==0, 0 )
        
        return torch.sum(result, dim=1) / (torch.sum(pooling_mask, dim=1) + 1e-9)
    
    

class FullAgentWisePooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, memory, mask=None):
        """
        memory.shape = (batch_size, length, num_tracks, d_model)
        mask : source mask
             shape = (batch_size, length * num_tracks or 1, length * num_tracks)
        """
        if mask is None:
            result = torch.mean(memory, dim=1)
            return torch.mean(result, dim=1)
        
        batch_size = memory.size(0)
        obs_length = memory.size(1)
        num_tracks = memory.size(2)
        
        if mask.size(-2) == 1:
            pooling_mask = mask.view(batch_size, obs_length, num_tracks)
        else:
            temporal_mask = get_temporal_mask(mask, obs_length, num_tracks) # shape = (batch_size, num_tracks, length, length)
            pooling_mask = ~(torch.sum( temporal_mask, dim=-2 ) == 0) # shape=(batch_size, num_tracks, length)
            pooling_mask = pooling_mask.transpose(1,2) # shape=(batch_size, length, num_tracks)
        
        #print(pooling_mask)
        pooling_mask = pooling_mask.unsqueeze(-1)
        
        result = memory.masked_fill( pooling_mask==0, 0 )
        result = torch.sum(result, dim=1) / (torch.sum(pooling_mask, dim=1) + 1e-9)  
        
        return torch.sum(result, dim=1)/torch.sum(pooling_mask[:,0], dim=1)
    
    
    
    
    