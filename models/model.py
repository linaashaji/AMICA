#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 18:38:32 2022

@author: lachaji
"""

import copy


from transformer.input_embedding import FixedTemporalEmbedding, FixedDeltaEmbedding
from transformer.temporal_attention import TemporalMultiHeadAttention
from transformer.position_wise_feedforward import PositionWiseFeedForward

from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer

import torch.nn as nn
import torch


class MultiIDModel(nn.Module):
    def __init__(self, ctx):
        super(MultiIDModel, self).__init__()

        input_max_len = ctx['input_max_len'] #list
        input_dim = ctx['input_dim'] #list
        delta_input_max_len = ctx['backprop_window']
        N = ctx['nlayer']
        d_model = ctx['d_model']
        d_ff = ctx['dff']
        num_heads = ctx['num_heads']
        dropout_rate = ctx['dropout_rate']
        pre_ln = False

        self.nb_ids = 10
        self.input_dim = input_dim
        
        c = copy.deepcopy
        
        temporal_attn = TemporalMultiHeadAttention(d_model, num_heads, dropout_rate)
        
        
        self.temporal_encoder_emb = nn.ModuleList([ FixedTemporalEmbedding(input_dim[i], d_model,
                                                                  dropout_rate, max_len=input_max_len[i]) for i in range(self.nb_ids)])
    
    
        
        transformer_feed_forward = PositionWiseFeedForward( (d_model, d_ff, d_model) , dropout=dropout_rate)
        


        self.temporal_encoder = nn.ModuleList([Encoder( EncoderLayer( d_model, c(temporal_attn),
                                                  c(transformer_feed_forward), dropout_rate, pre_ln ), N=N) for i in range(self.nb_ids)])
    
    
        self.delta_encoder_emb = FixedDeltaEmbedding(d_model, d_model,
                                                                  dropout_rate, max_len=delta_input_max_len)
        
        
        self.delta_temporal_encoder = Encoder( EncoderLayer( d_model, c(temporal_attn),
                                                  c(transformer_feed_forward), dropout_rate, pre_ln ), N=N)
    
    
        self.out_generator = nn.ModuleList([nn.Linear(d_model, input_dim[i]) for i in range(self.nb_ids)])
    
    def deltaoperation(self, x, xorig, mask_bert, xid):
        
        out = self.delta_encoder_emb(x, xid)
        out = self.delta_temporal_encoder(out, None)
        
        outf=out.flatten(0,1)
        xidf=xid.flatten(0,1)
        xorigf = xorig.flatten(0,1)
        mask_bertf = mask_bert.flatten(0,1)
        
        out_list, orig_list, maskbert_list = [], [], []
        for i in range(self.nb_ids):
            out_temp = self.out_generator[i](outf[xidf==i])
            orig_temp = xorigf[xidf==i][...,:self.input_dim[i]]
            mask_bert_temp = mask_bertf[xidf==i]
            
            if(out_temp.shape[0] == 0):
                out_list.append(None)
            else:
                out_list.append(out_temp)

            if(orig_temp.shape[0] == 0):
                orig_list.append(None)
                maskbert_list.append(None)
            else:
                orig_list.append(orig_temp)
                maskbert_list.append(mask_bert_temp)
                
        return out_list, orig_list, maskbert_list

    def forward(self, dictdata):
        
        vlist, slist, idlist, mblist = [], [], [], []
        for i in range(self.nb_ids): 
            values, steps, mask, mask_bert = dictdata[f'id{i+1}']
            values = self.temporal_encoder_emb[i](values)
            values = self.temporal_encoder[i](values, mask)
            ids = torch.zeros_like(steps) + i
            vlist.append(values)
            slist.append(steps)
            idlist.append(ids)
            mblist.append(mask_bert)
            
            
        vout = torch.cat(vlist, dim=1)
        sout = torch.cat(slist, dim=1)
        iout = torch.cat(idlist, dim=1)
        mbout = torch.cat(mblist, dim=1)
        
        return vout, sout, iout, mbout

