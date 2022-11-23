"""
Created on Feb 7

@author: lachaji

"""

import math

import torch
import torch.nn as nn



class FixedTemporalEmbedding(nn.Module):
    
    def __init__(self, input_dim, d_model=256, dropout = 0.1, max_len=690):
        super().__init__()   
        
        self.dropout = nn.Dropout(p=dropout)        
        self.w1 = nn.Linear(input_dim, d_model)  
        
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        div_term = torch.exp( torch.arange(0, d_model, 2) * 
                                -(math.log(10000.0) / d_model)
                            )

        pos_encoding[:, 0::2] = torch.sin( position * div_term )
        pos_encoding[:, 1::2] = torch.cos( position * div_term )
        pos_encoding = pos_encoding.unsqueeze(0)
        
        self.register_buffer('pos_encoding', pos_encoding)
        
    def forward(self, x):
        
        obs_length = x.size(1)
        batch_size = x.size(0)
        
        x = self.w1(x) 
        pos_enc = self.pos_encoding.repeat_interleave( batch_size, dim=0 )
        x = x + pos_enc[:, :obs_length]

        
        return self.dropout(x)


class FixedDeltaEmbedding(nn.Module):
    
    def __init__(self, input_dim, d_model=256, dropout = 0.1, max_len=250, max_ids=10):
        super().__init__()   
        
        self.dropout = nn.Dropout(p=dropout)        
        self.w1 = nn.Linear(input_dim, d_model)  
        
        #Temporal Positional Encoding
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        div_term = torch.exp( torch.arange(0, d_model, 2) * 
                                -(math.log(10000.0) / d_model)
                            )

        pos_encoding[:, 0::2] = torch.sin( position * div_term )
        pos_encoding[:, 1::2] = torch.cos( position * div_term )
        pos_encoding = pos_encoding.unsqueeze(0)
        
        
        #Agent Positional Encoding
        agent_encoding = torch.zeros(max_ids, d_model)
        agent_position = torch.arange(0, max_ids).unsqueeze(1)
        
        div_term = torch.exp( torch.arange(0, d_model, 2) * 
                                -(math.log(10000.0) / d_model)
                            )

        agent_encoding[:, 0::2] = torch.sin( agent_position * div_term )
        agent_encoding[:, 1::2] = torch.cos( agent_position * div_term )
        agent_encoding = agent_encoding.unsqueeze(0)
        
        self.register_buffer('pos_encoding', pos_encoding)
        self.register_buffer('agent_encoding', agent_encoding)
        
    def forward(self, x, xid):
        
        obs_length = x.size(1)
        batch_size = x.size(0)
        
        x = self.w1(x) 
        
        agent_encoding = self.agent_encoding.repeat_interleave( batch_size, dim=0 )
        pos_enc = self.pos_encoding.repeat_interleave( batch_size, dim=0 )
        x = x + pos_enc[:, :obs_length]
        
        
        xidf = xid.flatten(0,1).long()
        
        agent_enc = agent_encoding.flatten(0,1)[xidf].reshape(batch_size,obs_length,-1)  
        x = x + agent_enc
        
        return self.dropout(x)        

        
class OneInputEmbedding(nn.Module):

    """
    linear_projection( concat(linear projection, positional encoding) )
    See agentFormer paper : https://arxiv.org/pdf/2103.14023
    """
        
    def __init__(self, input_dim=2, d_model=256, dropout=0.1, max_len=12):
        super().__init__()
        #super(AgentFormerInputEmbedding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.w1 = nn.Linear(input_dim, d_model)
        self.w2 = nn.Linear(2*d_model, d_model)
        
        # Positional encoding
        # ... defined by Child classes
    
    def forward(self, x, start_time=0):
        """
        x.shape = (batch_size, obs_length, num_tracks, input_dim=2)
        start_time : int
            time of the first observation in x
        """
        batch_size = x.size(0)
        obs_length = x.size(1)
        d_model = x.size(2)
        
        max_len = self.pos_encoding.size(-2)
        assert obs_length + start_time <= max_len
        
        x = self.w1(x) # shape = (batch_size, obs_length, d_model)
        
        pos_enc = self.pos_encoding[:, start_time : start_time+obs_length]
        pos_enc = pos_enc.repeat_interleave( batch_size, dim=0 )
        # pos_enc.shape = (batch_size, obs_length, d_model)

        x = torch.cat( [ x,  pos_enc], dim=2 ) # shape = (batch_size, obs_length, 2 * d_model)
        x = self.w2(x) # shape = (batch_size, obs_length, d_model)
        return self.dropout(x)


class FixedOneInputEmbedding(OneInputEmbedding):
    def __init__(self, input_dim, d_model, dropout, max_len):
        super().__init__(input_dim=input_dim, d_model=d_model, dropout=dropout, max_len=max_len)

        # Positional encoding
        pos_encoding = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp( torch.arange(0, d_model, 2) * 
                                -(math.log(10000.0) / d_model)
                            )
        

        pos_encoding[:, 0::2] = torch.sin( position * div_term )
        pos_encoding[:, 1::2] = torch.cos( position * div_term )
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer('pos_encoding', pos_encoding)

class LearnedOneInputEmbedding(OneInputEmbedding):
    def __init__(self, input_dim, d_model, dropout, max_len):
        super().__init__(input_dim=input_dim, d_model=d_model, dropout=dropout, max_len=max_len)

        # Positional encoding
        pos_encoding = nn.Parameter( torch.randn(1, max_len, d_model)  / math.sqrt(d_model) , requires_grad = True )
        assert pos_encoding.shape == (1, max_len, d_model), f"pos_encoding.shape = {pos_encoding.shape} != {(1, max_len, d_model)}"
        
        # Save as a trainable param of the model
        self.register_parameter("pos_encoding", pos_encoding) 