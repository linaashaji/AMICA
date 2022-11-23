"""
Created on Feb 7

@author: lachaji

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask import get_temporal_mask, get_social_mask

#%%

class AxialMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, kind="TS"):
        """
        parameters
        ----------
        kind : str in ('TS', 'ST', 'P1', 'P2')
            TS : Temporal multi-head attention followed by Social multi-head attention
            ST : Social multi-head attention followed by Temporal multi-head attention
            P1 : mean of Social and Temporal
            #P2 : mean of TS and ST
        """
        super().__init__()
        assert d_model % num_heads == 0
        
        assert kind in ('TS', 'ST', 'P1', 'P2')
        
        if kind=='P2':
            raise NotImplementedError(" P2(Sum of TS and ST) is not implemented yet")
        
        self.kind = kind
        
        self.temporal_mha = TemporalMultiHeadAttention(d_model, num_heads, dropout, add_bias_kv=False, record_attn_weights=False)
        self.social_mha = SocialMultiHeadAttention(d_model, num_heads, dropout, add_bias_kv=False, record_attn_weights=False)
        
    
    def forward(self, query, key, value, mask=None, look_ahead=None):
        """
        Parameters
        ----------
        query :  Tensor[batch_size, query_length, num_tracks, d_model]
        key : Tensor[batch_size, memory_length, num_tracks, d_model]
        value : Tensor[batch_size, memory_length, num_tracks, d_model]
        mask : Tensor[batch_size, query_length*num_tracks, memory_length*num_tracks] or
                Tensor[batch_size, 1, memory_length*num_tracks] 
        """
        
        if self.kind == 'TS':
            # Temporal multi-head attention followed by Social multi-head attention
            temporal_output = self.temporal_mha(query, key, value, mask, look_ahead)
            #return  temporal_output + self.social_mha( temporal_output, key, value, mask)
            return self.social_mha( temporal_output, key, value, mask)
        
        elif self.kind == "ST":
            # Social multi-head attention followed by Temporal multi-head attention
            social_output = self.social_mha(query, key, value, mask)
            return self.temporal_mha( social_output, key, value, mask, look_ahead)
    
        elif self.kind == "P1":
            # Sum of Social and Temporal
            temporal_output = self.temporal_mha(query, key, value, mask, look_ahead)
            social_output = self.social_mha(query, key, value, mask)
            return temporal_output + social_output
        
        elif self.kind == "P2":
            pass
    
        else:
            assert self.kind in ('TS', 'ST', 'P1', 'P2')






class AxialMultiHeadAttentionLN(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, kind="TS"):
        """
        parameters
        ----------
        kind : str in ('TS', 'ST', 'P1', 'P2')
            TS : Temporal multi-head attention followed by Social multi-head attention
            ST : Social multi-head attention followed by Temporal multi-head attention
            P1 : mean of Social and Temporal
            #P2 : mean of TS and ST
        """
        super().__init__()
        assert d_model % num_heads == 0
        
        assert kind in ('TS', 'ST', 'P1', 'P2')
        
        if kind=='P2':
            raise NotImplementedError(" P2(Sum of TS and ST) is not implemented yet")
        
        self.kind = kind
        
        self.temporal_mha = TemporalMultiHeadAttention(d_model, num_heads, dropout, add_bias_kv=False, record_attn_weights=False)
        self.social_mha = SocialMultiHeadAttention(d_model, num_heads, dropout, add_bias_kv=False, record_attn_weights=False)
        self.norm_t = nn.LayerNorm(d_model)
        if(kind != 'P1'):
            self.norm_s = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None, look_ahead=None):
        """
        Parameters
        ----------
        query :  Tensor[batch_size, query_length, num_tracks, d_model]
        key : Tensor[batch_size, memory_length, num_tracks, d_model]
        value : Tensor[batch_size, memory_length, num_tracks, d_model]
        mask : Tensor[batch_size, query_length*num_tracks, memory_length*num_tracks] or
                Tensor[batch_size, 1, memory_length*num_tracks] 
        """
        
        if self.kind == 'TS':
            # Temporal multi-head attention followed by Social multi-head attention
            temporal_output = self.temporal_mha(query, key, value, mask, look_ahead)
            
            temporal_output = self.norm_t(self.dropout(temporal_output) + query)
            #return  temporal_output + self.social_mha( temporal_output, key, value, mask)
            
            social_output = self.social_mha(temporal_output, key, value, mask)
            
            return self.norm_s(self.dropout(social_output) + temporal_output)
        
        elif self.kind == "ST":
            # Social multi-head attention followed by Temporal multi-head attention
            social_output = self.social_mha(query, key, value, mask)
            social_output = self.norm_s(social_output + query)
            
            temporal_output = self.temporal_mha(social_output, key, value, mask, look_ahead)
            
            return self.norm_t(social_output + temporal_output)
    
        elif self.kind == "P1":
            # Sum of Social and Temporal
            temporal_output = self.temporal_mha(query, key, value, mask, look_ahead)
            social_output = self.social_mha(query, key, value, mask)
            return self.norm_t(social_output + temporal_output + query)
        
        elif self.kind == "P2":
            pass
    
        else:
            assert self.kind in ('TS', 'ST', 'P1', 'P2')



class TemporalMultiHeadAttentionLN(nn.Module):
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
        
        self.temporal_mha = TemporalMultiHeadAttention(d_model, num_heads, dropout,
                                                       add_bias_kv=add_bias_kv, record_attn_weights=record_attn_weights)
        self.norm_t = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None, look_ahead=None):
        """
        Parameters
        ----------
        query :  Tensor[batch_size, query_length, num_tracks, d_model]
        key : Tensor[batch_size, memory_length, num_tracks, d_model]
        value : Tensor[batch_size, memory_length, num_tracks, d_model]
        mask : Tensor[batch_size, query_length*num_tracks, memory_length*num_tracks] or
                Tensor[batch_size, 1, memory_length*num_tracks] 
        """
        
        temporal_output = self.temporal_mha(query, key, value, mask, look_ahead)
        
        temporal_output = self.norm_t(query + self.dropout(temporal_output))
        
        return temporal_output
        




def attn_order(x, attmaps):
    '''
        input:
            x.shape            --> (batch_size, timesteps, num_agents, d)
            attmaps.shape      --> (batch_size, num_heads, num_agents, timesteps, timesteps)
            
        output:
            xout.shape         --> (batch_size, timesteps, num_agents, d)
            inv_indices.shape  --> (batch_size, num_agents, timesteps)
    '''
    b, t, n, d = x.shape
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    p_attn = torch.mean(attmaps, dim = 1)
    p_attn_sum = torch.sum(p_attn, dim=-2)
    p_attn_sum_sorted, indices = torch.sort(p_attn_sum, descending = True)
    _, inv_indices = torch.sort(indices)
    
    findices = (indices + torch.arange(0, b*t*n, t).reshape(b,n,1).to(device)).flatten()
    xf = x.transpose(1,2).reshape(-1, d)
    xout = xf[findices].reshape(b,n,t,d).transpose(1,2)
    
    return xout, inv_indices


def attn_reorder(x, inv_indices):
    '''
        input:
            x.shape            --> (batch_size, timesteps, num_agents, d)
            inv_indices.shape  --> (batch_size, num_agents, timesteps)
            
        output:
            xout.shape         --> (batch_size, timesteps, num_agents, d)
    '''
    b, t, n, d = x.shape
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    findices = (inv_indices + torch.arange(0, b*t*n, t).reshape(b,n,1).to(device)).flatten()
    xf = x.transpose(1,2).reshape(-1, d)
    xout = xf[findices].reshape(b,n,t,d).transpose(1,2)
    
    return xout





class Order_AxialMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, attention_norm=False, dropout=0.1):
        """
        parameters
        ----------
        kind : TS : Temporal multi-head attention followed by Social multi-head attention
        """
        super().__init__()
        assert d_model % num_heads == 0
        
        self.attention_norm = attention_norm
        self.temporal_mha = TemporalMultiHeadAttention(d_model, num_heads, dropout, add_bias_kv=False, record_attn_weights=True)
        self.social_mha = SocialMultiHeadAttention(d_model, num_heads, dropout, add_bias_kv=False, record_attn_weights=False)
        if(attention_norm):
            self.norm_t = nn.LayerNorm(d_model)
            self.norm_s = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None, look_ahead=None):
        """
        Parameters
        ----------
        query :  Tensor[batch_size, query_length, num_tracks, d_model]
        key : Tensor[batch_size, memory_length, num_tracks, d_model]
        value : Tensor[batch_size, memory_length, num_tracks, d_model]
        mask : Tensor[batch_size, query_length*num_tracks, memory_length*num_tracks] or
                Tensor[batch_size, 1, memory_length*num_tracks] 
        """
               
        # Temporal multi-head attention followed by Social multi-head attention
        temporal_output = self.temporal_mha(query, key, value, mask, look_ahead)
        
        if(self.attention_norm):
            temporal_output = self.norm_t(self.dropout(temporal_output) + query)
        #return  temporal_output + self.social_mha( temporal_output, key, value, mask)
        
        attn_maps = self.temporal_mha.attention_weights
        
        order_temporal_output, inv_indices = attn_order(temporal_output, attn_maps)
        
        social_output = self.social_mha(order_temporal_output, order_temporal_output, order_temporal_output, mask)
        
        reorder_social_output = attn_reorder(social_output, inv_indices)
        
        if(self.attention_norm):
            return self.norm_s(self.dropout(reorder_social_output) + temporal_output)
        else:
            return self.dropout(reorder_social_output) + temporal_output






# ===============================================================================
#   TEMPORAL MULTI-HEAD ATTENTION
# ===============================================================================

    
def temporal_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled dot product attention over temporal dimension
    
    query : Tensor[batch_size, num_tracks, query_length, d_model]
    key : Tensor[batch_size, num_tracks, memory_length, d_model]
    value : Tensor[batch_size, num_tracks, memory_length, d_model]
    mask : Tensor[batch_size, num_tracks, num_head, query_length, memory_length]
    dropout : dropout rate
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # scores.shape = (batch_size, num_tracks, h, query_obs_length, key_obs_length)
    
    if mask is not None:
        #print(scores.shape, mask.shape)
        scores = scores.masked_fill(mask==0, -1e9)
    
    p_attn = F.softmax(scores, dim = -1)
    #if dropout is not None:
    #    p_attn = dropout(p_attn)
    
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
        x : Tensor[batch_size, num_tracks, length, d_model]

        Return
        x with shape (batch_size, num_tracks, num_heads, length, d_k)
        """
        return x.view(x.size(0), x.size(1), x.size(2), self.num_heads, self.d_k).transpose(-2, -3)
    
    def forward(self, query, key, value, mask=None, look_ahead=None):
        """
        Parameters
        ----------
        query :  Tensor[batch_size, query_length, num_tracks, d_model]
        key : Tensor[batch_size, memory_length, num_tracks, d_model]
        value : Tensor[batch_size, memory_length, num_tracks, d_model]
        mask : Tensor[batch_size, query_length*num_tracks, memory_length*num_tracks] or
                Tensor[batch_size, 1, memory_length*num_tracks] 
        """

        device = torch.get_device(query)
        device = "cpu" if device < 0 else device
        
        d_model = query.size(-1)
        num_tracks = query.size(-2)        
        query_obs_length = query.size(-3)
        
        assert key.size(-2) == num_tracks, "#pedestrians in the target trajectory != #pedestrians in source trajectory"
        memory_obs_length = key.size(-3)
        
        # Tranpose temporal and social dim
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)
        
        # Projection
        query = self.wq(query) # shape = (batch_size, num_tracks, memory_obs_length, d_model)
        key = self.wk(key)
        value = self.wv(value)
        
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        

        mask_to_apply = None
        if mask is not None and look_ahead is not None:
            temporal_mask = get_temporal_mask(mask, memory_obs_length, num_tracks)
            merged_mask = temporal_mask & look_ahead
            mask_to_apply = merged_mask.unsqueeze(2)
        elif mask is not None:
            mask_to_apply = get_temporal_mask(mask, memory_obs_length, num_tracks).unsqueeze(2)
        elif look_ahead is not None:
            mask_to_apply = look_ahead.unsqueeze(2)
            
            
        # temporal attention
        x, attention_weights = temporal_dot_product_attention(
            query, key,
            value,
            mask_to_apply,
            self.dropout
        ) # shape (batch_size, num_tracks, num_head, memory_length, d_k)
        
        if self.record_attn_weights:
            self.attention_weights = attention_weights.transpose(1, 2)
            # shape = (batch_size, num_head, num_tracks, query_length, key_length)
        
        x = x.transpose(-2, -3).contiguous().flatten(-2,-1)
        #.view(batch_size, num_tracks, memory_length, self.num_heads * self.d_k)
        # ReTranspose temporal and social dim
        x = x.transpose(1,2)
        return self.dense(x)




# ===============================================================================
#   SOCIAL MULTI-HEAD ATTENTION
# ===============================================================================


def social_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled dot product attention over social dimension
    
    query : Tensor[batch_size,length, num_tracks, d_model]
    key : Tensor[batch_size, length, num_tracks, d_model]
    value : Tensor[batch_size, length, num_tracks, d_model]
    mask : Tensor[batch_size, num_head, length, num_tracks, nul_tracks]
    dropout : dropout rate
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # scores.shape = (batch_size, query_obs_length, h, num_tracks, num_tracks)
    
    if mask is not None:
        #print(scores.shape, mask.shape)
        scores = scores.masked_fill(mask==0, -1e9)
    
    p_attn = F.softmax(scores, dim = -1)
    #if dropout is not None:
    #    p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn


class SocialMultiHeadAttention(nn.Module):
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
        x : Tensor[batch_size, length, num_tracks, d_model]

        Return
        x with shape (batch_size, length, num_heads, num_tracks, d_k)
        """
        return x.view(x.size(0), x.size(1), x.size(2), self.num_heads, self.d_k).transpose(-2, -3)
    
    def forward(self, query, key, value, mask=None, look_ahead=None):
        """
        Parameters
        ----------
        query :  Tensor[batch_size, query_length, num_tracks, d_model]
        key : Tensor[batch_size, memory_length, num_tracks, d_model]
        value : Tensor[batch_size, memory_length, num_tracks, d_model]
        mask : Tensor[batch_size, query_length*num_tracks, memory_length*num_tracks] or
                Tensor[batch_size, 1, memory_length*num_tracks] 
        """

        device = torch.get_device(query)
        device = "cpu" if device < 0 else device
        
        num_tracks = query.size(-2)        
        query_obs_length = query.size(-3)
        
        assert query.size(-2) == num_tracks, "#pedestrians in the target trajectory != #pedestrians in source trajectory"
        memory_obs_length = key.size(-3)
        
        assert query_obs_length == memory_obs_length or memory_obs_length == 1, "cannot do 'cross social attention' because there is no 1-1 corespondance in the temporal domain"
        
        
        # Projection
        query = self.wq(query) # shape = (batch_size, memory_length, num_tracks, d_model)
        key = self.wk(key)
        value = self.wv(value)
        
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        mask_to_apply = None
        if mask is not None:
            mask_to_apply = get_social_mask(mask, memory_obs_length, num_tracks).unsqueeze(2) # unsqueeze -> same mask applied to all h heads
        
        # social attention
        x, attention_weights = social_dot_product_attention(
            query, key,
            value,
            mask_to_apply,
            self.dropout
        ) # shape (bs, memory_obs_length, h, num_tracks, d_k)
        
        if self. record_attn_weights:
            self.attention_weights = attention_weights.transpose(1, 2)
            # shape = (batch_size, h, query_length, query_tracks, num_tracks)
        
        x = x.transpose(-2, -3).contiguous().flatten(-2,-1)#.view(batch_size, -1, self.num_heads * self.d_k)
        
        return self.dense(x)