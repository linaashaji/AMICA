"""
Created on Feb 7

@author: lachaji

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask import get_temporal_mask, get_social_mask



class AxialMultiHeadMixAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, kind="TS", record_attn_weights=False):
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
        
        self.temporal_mha = TemporalMixMultiHeadAttention(d_model, num_heads, dropout, add_bias_kv=False, record_attn_weights=record_attn_weights)
        self.social_mha = SocialMultiHeadAttention(d_model, num_heads, dropout, add_bias_kv=False, record_attn_weights=record_attn_weights)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, seed_length, src_mask=None, src_target_mask=None, mix_mask = None):
        """
        Parameters
        ----------
        query :  Tensor[batch_size, (seed_length + target_length), num_tracks, d_model]
        key : Tensor[batch_size, (seed_length + target_length), num_tracks, d_model]
        value : Tensor[batch_size, (seed_length + target_length), num_tracks, d_model]
        src_mask : Tensor[batch_size, 1, seed_length*num_tracks] 
        src_target_mask : Tensor[batch_size, 1, (seed_length + target_length) * num_tracks]
        mix_mask : Tensor[batch_size, num_tracks, target_length, (seed_length + target_length)]
        """
        
        if self.kind == 'TS':
            # Temporal multi-head attention followed by Social multi-head attention
            temporal_output = self.temporal_mha(query, key, value,
                                                seed_length, src_mask, src_target_mask, mix_mask)
            
            temporal_output = self.norm_t(temporal_output + query)
            #return  temporal_output + self.social_mha( temporal_output, key, value, mask)
            
            social_output = self.social_mha(temporal_output, key, value, src_target_mask)
            
            return self.norm_s(social_output + temporal_output)
        
        elif self.kind == "ST":
            # Social multi-head attention followed by Temporal multi-head attention
            social_output = self.social_mha(query, key, value, src_target_mask)
            social_output = self.norm_s(social_output + query)
            
            temporal_output = self.temporal_mha(social_output, key, value,
                                                seed_length, src_mask, src_target_mask, mix_mask)
            
            return self.norm_t(social_output + temporal_output)
    
        elif self.kind == "P1":
            # Sum of Social and Temporal
            temporal_output = self.temporal_mha(query, key, value,
                                                seed_length, src_mask, src_target_mask, mix_mask)
            social_output = self.social_mha(query, key, value, src_target_mask)
            return self.norm_t(social_output + temporal_output + query)
        
        elif self.kind == "P2":
            pass
    
        else:
            assert self.kind in ('TS', 'ST', 'P1', 'P2')




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




class TemporalMixMultiHeadAttention(nn.Module):
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
        
        self.attention_weights_mix = None # weight of the last forward pass 
        self.attention_weights_seed = None
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
    
    def forward(self, query, key, value, seed_length, src_mask=None, src_target_mask=None, mix_mask = None):
        """
        Parameters
        ----------
        query :  Tensor[batch_size, (seed_length + target_length), num_tracks, d_model]
        key : Tensor[batch_size, (seed_length + target_length), num_tracks, d_model]
        value : Tensor[batch_size, (seed_length + target_length), num_tracks, d_model]
        src_mask : Tensor[batch_size, 1, seed_length*num_tracks] 
        src_target_mask : Tensor[batch_size, 1, (seed_length + target_length) * num_tracks]
        mix_mask : Tensor[batch_size, num_tracks, target_length, (seed_length + target_length)]
        """

        num_tracks = query.size(-2) 
        seed_target_length = query.size(1) 
        
        assert key.size(-2) == num_tracks, "#pedestrians in the target trajectory != #pedestrians in source trajectory"
        
        # Tranpose temporal and social dim
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)
        
        # Shared Projection
        query = self.wq(query) # shape = (batch_size, num_tracks, memory_obs_length, d_model)
        key = self.wk(key)
        value = self.wv(value)
        

        #Self Attention
        seed_query = self.split_heads(query[...,:seed_length,:])
        seed_key = self.split_heads(key[...,:seed_length,:])
        seed_value = self.split_heads(value[...,:seed_length,:])        
        


        mask_to_apply = None
        if(src_mask is not None):
            mask_to_apply = get_temporal_mask(src_mask, seed_length, num_tracks).unsqueeze(2)
   
        x_seed, attention_weights_seed = temporal_dot_product_attention(
            seed_query, seed_key,
            seed_value,
            mask_to_apply,
            self.dropout
        )
        
        if self.record_attn_weights:
            self.attention_weights_seed = attention_weights_seed.transpose(1, 2)
            
        # Concat
        x_seed = x_seed.transpose(-2, -3).contiguous().flatten(-2,-1)
        x_seed = x_seed.transpose(1,2) 
       
       
   
        target_query = self.split_heads(query[...,seed_length:,:])
        mix_key = self.split_heads(key)
        mix_value = self.split_heads(value)   


        mask_to_apply = None
        if src_target_mask is not None and mix_mask is not None:
            temporal_mask = get_temporal_mask(src_target_mask, seed_target_length, num_tracks)
            merged_mask = temporal_mask & mix_mask
            mask_to_apply = merged_mask.unsqueeze(2)
        elif src_target_mask is not None:
            mask_to_apply = get_temporal_mask(src_target_mask, seed_target_length, num_tracks).unsqueeze(2)
        elif mix_mask is not None:
            mask_to_apply = mix_mask.unsqueeze(2)
            
            
            
        x_target, attention_weights_mix = temporal_dot_product_attention(
            target_query, mix_key,
            mix_value,
            mask_to_apply,
            self.dropout
        )  

        if self.record_attn_weights:
            self.attention_weights_mix = attention_weights_mix.transpose(1, 2)
            
        x_target = x_target.transpose(-2, -3).contiguous().flatten(-2,-1)
        x_target = x_target.transpose(1,2)
        
        x = torch.cat((x_seed,x_target), dim=1)

        return self.dense(x)
    
    
    
    
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
    
    def forward(self, query, key, value, mask=None):
        """
        Parameters
        ----------
        query :  Tensor[batch_size, (seed_length + target_length), num_tracks, d_model]
        key : Tensor[batch_size, (seed_length + target_length), num_tracks, d_model]
        value : Tensor[batch_size, (seed_length + target_length), num_tracks, d_model]
        mask : Tensor[batch_size, 1, (seed_length + target_length)*num_tracks] 
        """
        
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
        )
        
        if self. record_attn_weights:
            self.attention_weights = attention_weights.transpose(1, 2)
            # shape = (batch_size, h, query_length, query_tracks, num_tracks)
        
        x = x.transpose(-2, -3).contiguous().flatten(-2,-1)#.view(batch_size, -1, self.num_heads * self.d_k)
        
        return self.dense(x)























