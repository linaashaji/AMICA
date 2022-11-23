"""
Created on Feb 7

@author: lachaji

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask import get_1d_temporal_mask

def temporal_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot product attention over temporal dimension
    
    query : Tensor[batch_size, query_length, d_model]
    key : Tensor[batch_size, memory_length, d_model]
    value : Tensor[batch_size, memory_length, d_model]
    mask : Tensor[batch_size, num_head, query_length, memory_length]
    dropout : dropout rate
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # scores.shape = (batch_size, h, query_obs_length, key_obs_length)
    
    if mask is not None:

        scores = scores.masked_fill(mask==0, -1e9)
    
    p_attn = F.softmax(scores, dim = -1)
    
    return torch.matmul(p_attn, value), p_attn



class TemporalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, add_bias_kv=False, record_attn_weights=False):
        """
        Parameters
        ----------
        d_model : int
            embedding dimension
        num_heads : real
            number of heads
        dropout : float
             in [0.0, 1.0)
        add_bias_kv : bool
            whether to use bias in the linear projection of key and value
        record_attn_weights : bool
            whether to record attention weights during forward pass

        """

        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=add_bias_kv)
        
        self.wv = nn.Linear(d_model, d_model, bias=add_bias_kv)
        self.dense = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.attention_weights = None # weight of the last forward pass 
        self.record_attn_weights = record_attn_weights
    
    def split_heads(self, x):
        """
        Parameters
        ----------
        x : Tensor[batch_size, length, d_model]

        Return
        x with shape (batch_size, length, num_heads, d_k)
        """
        return x.view(x.size(0), x.size(1), self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, query, key, value, mask=None, look_ahead_mask = None):
        """
        Parameters
        ----------
        query :  Tensor[batch_size, query_length, d_model]
        key : Tensor[batch_size, memory_length, d_model]
        value : Tensor[batch_size, memory_length, d_model]
        mask : Tensor[batch_size, 1, memory_length] 
        """

        device = torch.get_device(query)
        device = "cpu" if device < 0 else device
        memory_obs_length = key.size(-2)
        query_obs_length = query.size(-2)
        
        # Projection
        query = self.wq(query) # shape = (batch_size, memory_obs_length, d_model)
        key = self.wk(key)
        value = self.wv(value)
        
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        mask_to_apply = None
        if mask is not None and look_ahead_mask is not None:
            merged_mask = mask & look_ahead_mask
            mask_to_apply = get_1d_temporal_mask(merged_mask, memory_obs_length)
            mask_to_apply = mask_to_apply[..., :query_obs_length,:]
        elif mask is not None:
            mask_to_apply = get_1d_temporal_mask(mask, memory_obs_length)
        elif look_ahead_mask is not None:
            mask_to_apply = get_1d_temporal_mask(look_ahead_mask, memory_obs_length)
            mask_to_apply = mask_to_apply[..., :query_obs_length,:]
        
        # temporal attention
        x, attention_weights = temporal_dot_product_attention(
            query, key,
            value,
            mask_to_apply
        ) # shape (batch_size, num_tracks, num_head, memory_length, d_k)
        
        if self.record_attn_weights:
            self.attention_weights = attention_weights.transpose(1, 2)
            # shape = (batch_size, num_head, query_length, key_length)
            
        # Concat
        batch_size = query.size(0)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.dense(x)