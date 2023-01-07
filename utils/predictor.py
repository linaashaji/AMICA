# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 22:41:50 2023

@author: U629826
"""


import time
import os
import torch
from .timer import Timer
from .utils import print_log, batch_to_device
import numpy as np

timer = Timer()



def run_statistics(model, log, data_iter, data_ctx, loss_compute, device, verbose=50):

    model.eval()
    losses = []

    window_size = data_ctx['window']
    backprop_window = data_ctx['backprop_window']
    nb_ids = 10
    
    for i, batch in enumerate(data_iter):
     
        batch = batch_to_device(batch, device) 
        batch_size = batch['id1'][0].size(0)
        
        vorig_list = []
        for k in range(nb_ids): 
            valuesorig = batch[f'orig_id{k+1}']

            vorig_list.append(valuesorig)

        vorig = torch.cat(vorig_list, dim=1).to(device)

        #vout: masked values --> shape= (batch_size, id_seq_len, D)
        #sout: steps: unordered global steps for the id --> shape= (batch_size, id_seq_len) 
        #iout: ids: the id for each step --> shape= (batch_size, id_seq_len) 
        #mbout: bert mask: True for masked steps --> shape= (batch_size, id_seq_len) 
        
        vout, sout, iout, mbout = model(batch)
    
        _, indices = torch.sort(sout)
        
        ind = indices + torch.arange(batch_size).unsqueeze(1).to(device) * indices.shape[-1]
        ind = ind[:, :window_size].flatten(0,1)
        
        vsorted = vout.flatten(0,1)[ind].reshape(batch_size,window_size,-1) 
        
        vorigsorted = vorig.flatten(0,1)[ind].reshape(batch_size,window_size,-1)  
        
        isorted = iout.flatten(0,1)[ind].reshape(batch_size,window_size)   
        
        mbsorted = mbout.flatten(0,1)[ind].reshape(batch_size,window_size)
        
        
        
        for j in range(0, window_size, backprop_window):
            
            #Update Spatial contextual representation
            vinp = vsorted[:, j: j+backprop_window]
            voriginp = vorigsorted[:, j: j+backprop_window]
            iinp = isorted[:, j: j+backprop_window]
            mbinp = mbsorted[:, j: j+backprop_window]

            vdeltaout, vtarget, mbout, bout = model.deltaoperation(vinp, voriginp, mbinp, iinp)
            

            loss_dict = loss_compute(vdeltaout, vtarget, mbout, batch_size, bout)  
            
            #Update temporal contextual representation
            vout, _, _, _ = model(batch)
            vsorted = vout.flatten(0,1)[ind].reshape(batch_size,window_size,-1) 
            

            losses.extend(list(loss_dict['loss'].detach().cpu().numpy()))


        if verbose > 0:
            if (i+1) % verbose == 0:
                print_log(" Update step : %d " % i, log)

        
    return np.array(losses)





def run_test(model, log, data_iter, data_ctx, loss_compute, threshold, device, verbose=50):

    model.eval()

    window_size = data_ctx['window']
    backprop_window = data_ctx['backprop_window']
    nb_ids = 10
    
    predicted_labels = []
    gt_labels = []
    
    total_count = 0
    
    for i, batch in enumerate(data_iter):
        
        total_count +=1
     
        batch = batch_to_device(batch, device) 
        batch_size = batch['id1'][0].size(0)
        
        vorig_list = []
        for k in range(nb_ids): 
            valuesorig = batch[f'orig_id{k+1}']

            vorig_list.append(valuesorig)

        vorig = torch.cat(vorig_list, dim=1).to(device)

        #vout: masked values --> shape= (batch_size, id_seq_len, D)
        #sout: steps: unordered global steps for the id --> shape= (batch_size, id_seq_len) 
        #iout: ids: the id for each step --> shape= (batch_size, id_seq_len) 
        #mbout: bert mask: True for masked steps --> shape= (batch_size, id_seq_len) 
        
        vout, sout, iout, mbout = model(batch)
    
        _, indices = torch.sort(sout)
        
        ind = indices + torch.arange(batch_size).unsqueeze(1).to(device) * indices.shape[-1]
        ind = ind[:, :window_size].flatten(0,1)
        
        vsorted = vout.flatten(0,1)[ind].reshape(batch_size,window_size,-1) 
        
        vorigsorted = vorig.flatten(0,1)[ind].reshape(batch_size,window_size,-1)  
        
        isorted = iout.flatten(0,1)[ind].reshape(batch_size,window_size)   
        
        mbsorted = mbout.flatten(0,1)[ind].reshape(batch_size,window_size)
        
        
        batch_loss = torch.zeros(batch_size, window_size // backprop_window)
        k = 0
        for j in range(0, window_size, backprop_window):
            
            #Update Spatial contextual representation
            vinp = vsorted[:, j: j+backprop_window]
            voriginp = vorigsorted[:, j: j+backprop_window]
            iinp = isorted[:, j: j+backprop_window]
            mbinp = mbsorted[:, j: j+backprop_window]

            vdeltaout, vtarget, mbout, bout = model.deltaoperation(vinp, voriginp, mbinp, iinp)
            

            loss_dict = loss_compute(vdeltaout, vtarget, mbout, batch_size, bout)  
            
            #Update temporal contextual representation
            vout, _, _, _ = model(batch)
            vsorted = vout.flatten(0,1)[ind].reshape(batch_size,window_size,-1) 
            
            batch_loss[:, k] = loss_dict['loss'].detach()
            k +=1
            
            
        for bl, gt in zip(batch_loss, batch['label']):
            r = bl - threshold
            if((r>0).any()):
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
                
            gt_labels.append(int(gt))
            

        
    return predicted_labels, gt_labels