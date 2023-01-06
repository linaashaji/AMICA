"""
Created on Feb 7

@author: lachaji

"""

import time
import os
import torch
from .timer import Timer
from .utils import print_log, batch_to_device
import numpy as np

timer = Timer()


def run_epoch(model, log, data_iter, data_ctx, loss_compute, device, epoch, split, writer, verbose=50):

    
    start = time.time()
    timer = Timer()
    timer.tik()
    nb_scenes = 0
    total_scenes = 0

    total_loss_dict = None
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
            

            loss_dict = loss_compute(vdeltaout, vtarget, mbout)    
            
            #Update temporal contextual representation
            vout, _, _, _ = model(batch)
            vsorted = vout.flatten(0,1)[ind].reshape(batch_size,window_size,-1) 
            
                
            if total_loss_dict is None:
                total_loss_dict = dict()
                for loss_name in loss_dict.keys():
                    total_loss_dict[loss_name] = loss_dict[loss_name].detach().item() 
            else:
                for loss_name in total_loss_dict.keys():
                    total_loss_dict[loss_name] += loss_dict[loss_name].detach().item() 
                

        total_scenes +=1
        nb_scenes +=batch_size
        
        if writer is not None:

            for loss_name in total_loss_dict.keys():
                writer.global_step +=1
                writer.add_scalar(f'step_{loss_name.capitalize()}/{split}',
                    total_loss_dict[loss_name],
                    writer.global_step)
                
            
        if verbose > 0:
            if (i+1) % verbose == 0:
                elapsed = time.time() - start
                print_log(" Update step : %d Loss : %f samples per Sec : %f " %
                      ( i, total_loss_dict['total_loss'] / total_scenes,
                       nb_scenes // elapsed ), log)
                start = time.time()
                nb_scenes = 0
                
    for loss_name in total_loss_dict.keys():
        total_loss_dict[loss_name] /= total_scenes     
        
    return {
        "Loss" : total_loss_dict['total_loss'],
        "loss_dict" : total_loss_dict,
        "duration" : timer.tak()
    }




def train_epoch(model, data_ctx, log, loss_compute_train, loss_compute_val, train_loader, val_loader, test_loader, epoch, writer,
          device, verbose, run_epoch_func):
    
    
    scores = {
        "train" : [],
        "val" : [],
        "test" : []
    }

    def train_step(split, loader, loss_compute):
        assert split in ('train', 'val', 'test')
            

        epoch_scores =  run_epoch_func(model, log, loader, data_ctx, loss_compute, device,
                                             epoch, split, writer, verbose=verbose)

        epoch_scores['epoch'] = epoch
        scores[split].append( epoch_scores )

        if writer is not None:

            for loss_name in epoch_scores['loss_dict'].keys():
                writer.add_scalar(f'epoch_{loss_name.capitalize()}/{split}',
                    epoch_scores['loss_dict'][loss_name],
                    epoch + 1)
            


        split = split.capitalize()

        txt = f"Epoch {epoch}: | {split}_Loss { epoch_scores['Loss'] }"
        print_log(txt, log)
                 
        

    model.train()
    train_step("train", train_loader, loss_compute_train)
    
    model.eval()
    train_step("val", val_loader, loss_compute_val)

    if test_loader:
        train_step("test", test_loader, loss_compute_val)

    print_log("-", log)

    return scores



def train(model, model_ctx, data_ctx, training_info, cfg, log, model_opt, loss_compute_train, loss_compute_val,
          train_loader, val_loader, test_loader, epoch_start, n_epochs, writer,
          device, verbose=-1, run_epoch_func=run_epoch, metrics = ['Loss']):
    
    print_log('\n', log)
    print_log(" Start training ".center(70, "="), log)


    timer.tik()

    n_epochs = cfg.num_epochs
    minmetrics = metrics[:]
    for i in range(len(minmetrics)):
        minmetrics[i] = np.inf
        
    metrics_val = minmetrics[:]
    
    save_model = False
    
    
    for ep_start in range(epoch_start, n_epochs):
        
        scores = train_epoch(model, data_ctx, log, loss_compute_train, loss_compute_val, train_loader, val_loader, test_loader,
                             ep_start, writer, device, verbose, run_epoch_func)

        if(ep_start % cfg.model_save_freq == 0):
            save_model = True
            best = ''
            
        for i in range(len(minmetrics)):
            if(test_loader):
                metrics_val[i] = float(scores['test'][0][f'{metrics[i]}'])
            else:
                metrics_val[i] = float(scores['val'][0][f'{metrics[i]}'])

            if(metrics_val[i] < minmetrics[i]):
                minmetrics[i] = metrics_val[i]
                save_model = True
                best = '_best'


        if(save_model == True):
            print_log(f"=== Saving model at epoch {ep_start}", log)
            checkpoint_filepath = cfg.model_path

            if not os.path.exists(checkpoint_filepath):
                os.mkdir(checkpoint_filepath)
    
    
            training_info['epoch'] = ep_start
            
            training_info["training_scores"]['train'] += scores['train']
            training_info["training_scores"]['val'] += scores['val']
            if(test_loader):
                training_info["training_scores"]['test'] += scores['test']
    

            model_state_dict = model.state_dict()
            try:
                optimizer_state_dict = model_opt.optimizer.state_dict()
            except:
                optimizer_state_dict = model_opt.state_dict()
                
            torch.save({
                "model_params" : model_ctx,
                "model_state_dict" : model_state_dict,
                "optimizer_state_dict" : optimizer_state_dict,
                "scheduler_state_dict" : None,
                "training_info" : training_info
            }, checkpoint_filepath + f"/model_{training_info['epoch']}{best}.pt" )
    
        save_model = False
        

    print_log(f"Time spend : {timer.tak()}", log)

    
