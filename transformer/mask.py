"""
Created on Feb 7

@author: lachaji

"""

import numpy as np

import torch
#import torch.nn as nn

def get_look_ahead_mask(size):
    """
    size : int
    return look ahead mask of shape (1, size, size)
    """
    attn_shape = (1, size, size)
    mask = torch.triu( torch.ones( attn_shape ), diagonal = 1)
    return mask == 0


def get_multi_look_ahead_mask(size, num_tracks):
    """
    size : int
    return look ahead mask of shape (num_tracks, size, size)
    """
    attn_shape = ( num_tracks, size, size)
    mask = torch.triu( torch.ones( attn_shape ), diagonal = 1)
    return mask == 0



def get_mix_mask(size, seed_length):
    """
    size : int
    return mix mask of shape (1, size - seed_length, size)
    """
    
    n = seed_length
    m = size - seed_length
    
    
    attn_shape = (1, m, m)
    mask_m = torch.triu( torch.ones( attn_shape ), diagonal = 1)
    
    mask_n = torch.zeros((1, m, n))
    
    mask = torch.cat((mask_n, mask_m), dim = 2)
    
    return mask == 0

def get_multi_mix_mask(size, seed_length, num_tracks):
    """
    size : int
    return mix mask of shape (1, size - seed_length, size)
    """
    
    n = seed_length
    m = size - seed_length
    
    
    attn_shape = (num_tracks, m, m)
    mask_m = torch.triu( torch.ones( attn_shape ), diagonal = 1)
    
    mask_n = torch.zeros((num_tracks, m, n))
    
    mask = torch.cat((mask_n, mask_m), dim = 2)
    
    return mask == 0


def get_goal_look_ahead_mask(size):
    """
    size : int
    return look ahead mask of shape (1, size, size)
    """
    attn_shape = (1, size, size)
    mask = torch.triu( torch.ones( attn_shape ), diagonal = 1)
    mask = 1 - mask
    
    return mask == 0

def get_multi_agent_subsequent_mask(obs_length, num_tracks=1, time_first=True, multi_agent_teacher_forcing=False):
    """
    Return mask of shape (1, length*num_tracks, length*num_tracks) where the timestep t of each agent can attend to the timestep
    of all the agent up to t-1 or up to timestep t in the case where "multi_agent_teacher_forcing" is set to True.
    
    time_fist : bool
        if True the return shape is (1, length*num_tracks, length*num_tracks) else (1, num_tracks*length, num_tracks*length)
    """
    def mask_function(i, j):
        mask = i//num_tracks <= j//num_tracks
        if multi_agent_teacher_forcing:
            mask = mask | ( (i//num_tracks - j//num_tracks == 1) & ( (i-j)%num_tracks!=0 ) )
        return ~mask
    
    size = num_tracks * obs_length
    attn_shape = (1, size, size)
    
    coords = np.meshgrid( np.arange(size), np.arange(size) )
    
    multi_agent_subsequent_mask = mask_function(coords[0], coords[1]).reshape(attn_shape).astype('uint8')
    mask = torch.from_numpy( multi_agent_subsequent_mask ) == 0
    if time_first:
        return mask
    else:
        mask = mask.reshape(obs_length, num_tracks, obs_length, num_tracks)
        mask = mask.transpose(0,1).transpose(-2,-1).flatten(0,1).flatten(-2, -1)
        return mask.unsqueeze(0)

def get_self_agent_mask(obs_length_memory, obs_length_query, num_tracks=1, time_first=True):
    """
        return mask of shape (1, obs_length_query*num_tracks, obs_length_memory*num_tracks) where each timestep of 
        each agent is allowed to access only to timestep of its own trajectory

    """
    def mask_function(i, j):
            return (i - j) % num_tracks != 0

    size_memory = obs_length_memory * num_tracks
    size_query = obs_length_query * num_tracks
    
    attn_shape = (1, size_query, size_memory)

    coords = np.meshgrid( np.arange(size_memory), np.arange(size_query) )

    self_agent_mask = mask_function(coords[0], coords[1]).reshape(attn_shape).astype('uint8')
    mask = torch.from_numpy( self_agent_mask ) == 0
    
    if time_first:
        return mask
    else:
        mask = mask.reshape(obs_length_query, num_tracks, obs_length_memory, num_tracks)
        mask = mask.transpose(0,1).transpose(-2,-1).flatten(0,1).flatten(-2, -1)
        return mask.unsqueeze(0)



def get_social_mask(multi_agent_mask, obs_length, num_tracks):
    """
    Extract social mask from the joint socio-temporal mask "multi_agent_mask"
    The social mask indicates at eachs timestep which agent can attend to each agent.

    Parameters
    ----------
    multi_agent_mask : Tensor[..., length_query*num_tracks, length_memory*num_tracks] or
                       Tensor[..., 1, length_memory*num_tracks]

    obs_length : temporal length of the query.
    num_tracks : number of pedestrians.

    return
    ------
    Tensor[..., length_query, num_tracks, num_tracks]
    """
    
    if multi_agent_mask is None :
        #print(multi_agent_mask)
        return None

    
    #assert multi_agent_mask.shape[-1] == obs_length*num_tracks
    assert multi_agent_mask.shape[-2] in (1, obs_length*num_tracks)
    
    if multi_agent_mask.shape[-2] > 1:
        assert multi_agent_mask.shape[-2] % num_tracks == 0
        query_obs_length = multi_agent_mask.shape[-2] // num_tracks
        mask = multi_agent_mask.reshape((-1, query_obs_length, num_tracks, obs_length, num_tracks))

        mask = mask[..., torch.arange(obs_length), :, torch.arange(obs_length), :]
    else:
        mask = multi_agent_mask.reshape((-1, 1, 1, obs_length, num_tracks))
        mask = mask[..., torch.arange(1), :, torch.arange(obs_length), :]
    
    #print(multi_agent_mask.shape, mask.shape)
    return mask.transpose(0,1)

def get_temporal_mask(multi_agent_mask, obs_length, num_tracks):
    """
    Extract temporal mask from the joint socio-temporal mask "multi_agent_mask"
    
    Parameters
    ----------
    multi_agent_mask : Tensor[..., length_query*num_tracks, length_memory*num_tracks] or
                       Tensor[..., 1, length_memory*num_tracks]

    obs_length : temporal length of the query.
    num_tracks : number of pedestrians.

    return
    ------
    Tensor[..., num_tracks, length, length]
    """

    if multi_agent_mask is None :
        return None

    
    #assert multi_agent_mask.shape[-1] == obs_length*num_tracks
    assert multi_agent_mask.shape[-2] in (1, obs_length*num_tracks)

    if multi_agent_mask.shape[-2] > 1:
        assert multi_agent_mask.shape[-2] % num_tracks == 0
        query_obs_length = multi_agent_mask.shape[-2] // num_tracks
        mask = multi_agent_mask.reshape((-1, query_obs_length, num_tracks, obs_length, num_tracks))

        mask = mask[..., :, torch.arange(num_tracks), :, torch.arange(num_tracks)]
    else:
        mask = multi_agent_mask.reshape((-1, 1, 1, obs_length, num_tracks))
        mask = mask[..., :, torch.arange(1), :, torch.arange(num_tracks)]

    #print(multi_agent_mask.shape, mask.shape)

    return mask.transpose(0,1)

def get_1d_temporal_mask(mask, obs_length):
    """
    Extract temporal mask from the joint socio-temporal mask "multi_agent_mask"
    
    Parameters
    ----------
    multi_agent_mask : Tensor[..., 1, length_memory*num_tracks]

    obs_length : temporal length of the query.

    return
    ------
    Tensor[..., length, length]
    """

    if mask is None :
        return None
    
    assert mask.shape[-2] in (1, obs_length) , mask.shape

    return mask.reshape((-1, 1, mask.shape[-2], obs_length))

def get_future_decoder_mask(mask, batch_size, length, num_tracks, pred_length):
    """
    Return a mask of shape (batch_size, 1, pred_length*num_tracks) that forbid access to agent with 
    all timestep missing (ex : padding pedestrian) based on the socio-temporal mask "mask"
    
    -> it creates a mask for the decoding part from the mask of the encoding part.

    Parameters
    ----------
    mask : Tensor[batch_size, length*num_tracks, length*num_tracks] or
                       Tensor[batch_size, 1, length*num_tracks]

    batch_size : int 
        =mask.shape[0]
    length : int
        number of timestep of the mask
    num_tracks : int
        number of pedestrians
    pred_length : int
        number of timestep of the return mask (set to the length of the predicted trajectory)

    """
        
    if mask.size(-2) == 1:
        future_decoder_mask = mask.view(batch_size, length, num_tracks)
    else:
        temporal_mask = get_temporal_mask(mask, length, num_tracks) # shape = (batch_size, num_tracks, obs_length, obs_length)
        future_decoder_mask = ~(torch.sum( temporal_mask, dim=-2 ) == 0) # shape=(batch_size, num_tracks, obs_length)
        future_decoder_mask = future_decoder_mask.transpose(1,2) # shape=(batch_size, obs_length, num_tracks)
    
    future_decoder_mask = future_decoder_mask.any(1, keepdim=True) # shape=(batch_size, 1, num_tracks)
    future_decoder_mask = future_decoder_mask.repeat_interleave(pred_length, dim=1).flatten(1, 2).unsqueeze(-2)
    # shape=(batch_size, 1, length*num_tracks)
    return future_decoder_mask




    