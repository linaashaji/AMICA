"""
Created on Feb 7

@author: lachaji

"""

import random
import numpy as np
import torch
import time
import os
import re
import matplotlib.pyplot as plt
import shutil

def get_state(model_name, device, directory="checkpoints/"):
    epoch_start = 0
    checkpoint_filepath = os.path.join(directory, model_name)
    #print(checkpoint_filepath)
    checkpoint = None

    if os.path.exists(checkpoint_filepath):
        files = os.listdir(checkpoint_filepath)
        #print(files)
        files = [file for file in files if re.match("model_epoch[0-9]+", file)]
        epochs = [int( re.findall("model_epoch([0-9]+).pt", file)[0] ) for file in files]
        
        files = sorted( list(zip(files, epochs)), key = lambda x : -x[1]) 
        
        #print(files)
        
        if len(files) > 0:
            file, epoch_start = files[0]
            print("=== The model exists")
            print("      Loading of the version correspoding to the highest epochs : ", f"<{file}> - epoch {epoch_start}")

            checkpoint = torch.load( os.path.join(checkpoint_filepath, file), map_location=device)
    
            return checkpoint
    
    print("=== The model doesn't exist")
    return None


def load_state(model, model_opt, model_name, device, directory="checkpoints/"):
    epoch_start = 0
    checkpoint_filepath = os.path.join(directory, model_name)
    checkpoint = None

    if os.path.exists(checkpoint_filepath):
        files = os.listdir(checkpoint_filepath)
        #print(files)
        files = [file for file in files if re.match("model_epoch[0-9]+", file)]
        epochs = [int( re.findall("model_epoch([0-9]+).pt", file)[0] ) for file in files]
        
        files = sorted( list(zip(files, epochs)), key = lambda x : -x[1]) 
        
        #print(files)
        
        if len(files) > 0:
            file, epoch_start = files[0]
            print("=== The model exists")
            print("      Loading of the version correspoding to the highest epochs : ", f"<{file}> - epoch {epoch_start}")

            checkpoint = torch.load( os.path.join(checkpoint_filepath, file), map_location=device)

            #model = AgentFormer( **checkpoint['model_params'] )
            print("    ", model.load_state_dict( checkpoint['model_state_dict'] ) )
            model.to(device)
            
            #print("    ", model_opt.scheduler.load_state_dict( checkpoint['scheduler_state_dict'] ) )
            print("    ", model_opt.optimizer.load_state_dict( checkpoint['optimizer_state_dict'] ) )
        
        else:
            print("=== The model doesn't exist : training from scratch")
    else:
        print("=== The model doesn't exist : training from scratch")
    return checkpoint


def load_state_best(model, model_opt, model_path, device):
    epoch_start = 0
    checkpoint_filepath = model_path
    checkpoint = None

    if os.path.exists(checkpoint_filepath):
        files = os.listdir(checkpoint_filepath)
        #print(files)
        files = [file for file in files if re.match("model_[0-9]+_best", file)]
        epochs = [int( re.findall("model_([0-9]+)_best.pt", file)[0] ) for file in files]
        
        files = sorted( list(zip(files, epochs)), key = lambda x : -x[1]) 
        
        #print(files)
        
        if len(files) > 0:
            file, epoch_start = files[0]
            print("=== The model exists")
            print("      Loading of the version correspoding to the highest epochs : ", f"<{file}> - epoch {epoch_start}")

            checkpoint = torch.load( os.path.join(checkpoint_filepath, file), map_location=device)
            
            try:
                print("    ", model.load_state_dict( checkpoint['model_state_dict'] ) )
                model.to(device)
            except:
                pretrained_dict = checkpoint['model_state_dict']
                model_dict = model.state_dict()
                pretrained_dict = {k: v if k in model_dict else print(f'Skipping {k}') for k, v in pretrained_dict.items()}
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                model.to(device)
                
        else:
            print("=== The model doesn't exist : training from scratch")
    else:
        print("=== The model doesn't exist : training from scratch")
    return checkpoint



def load_model_best(model_fn, model_path, device):
    epoch_start = 0
    checkpoint_filepath = model_path
    checkpoint = None

    if os.path.exists(checkpoint_filepath):
        files = os.listdir(checkpoint_filepath)
        #print(files)
        files = [file for file in files if re.match("model_[0-9]+_best", file)]
        epochs = [int( re.findall("model_([0-9]+)_best.pt", file)[0] ) for file in files]
        
        files = sorted( list(zip(files, epochs)), key = lambda x : -x[1]) 
        
        #print(files)
        
        if len(files) > 0:
            file, epoch_start = files[0]
            print("=== The model exists")
            print("      Loading of the version correspoding to the highest epochs : ", f"<{file}> - epoch {epoch_start}")

            checkpoint = torch.load( os.path.join(checkpoint_filepath, file), map_location=device)
            model_ctx = checkpoint['model_params']
            model = model_fn(model_ctx).to(device)
            try:
                print("    ", model.load_state_dict( checkpoint['model_state_dict'] ) )
                model.to(device)
            except:
                pretrained_dict = checkpoint['model_state_dict']
                model_dict = model.state_dict()
                pretrained_dict = {k: v if k in model_dict else print(f'Skipping {k}') for k, v in pretrained_dict.items()}
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                model.to(device)
                
            return model, checkpoint
                
        else:
            print("=== The model doesn't exist : training from scratch")
            return None, None
    else:
        print("=== The model doesn't exist : training from scratch")
        return None, None
    
        
        
def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def compute_grad_norm(params):
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    return norm


def nan_to_num(x, replace_nan_with):
    #print(x)
    x[x.isnan()] = replace_nan_with
    return x

def clamp_max(x, max_value):
    cst = torch.masked_fill(max_value/x, x<max_value, 1).detach()
    return x * cst


def print_log(print_str, log, same_line=False, display=True):
    '''
    print a string to a log file

    parameters:
        print_str:          a string to print
        log:                a opened file to save the log
        same_line:          True if we want to print the string without a new next line
        display:            False if we want to disable to print the string onto the terminal
    '''
    if display:
        if same_line: print('{}'.format(print_str), end='')
        else: print('{}'.format(print_str))

    if same_line: log.write('{}'.format(print_str))
    else: log.write('{}\n'.format(print_str))
    log.flush()   


def recreate_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d) 
  
    

def logging(cfg, epoch, total_epoch, iter, total_iter, ep, seq, frame, losses_str, log):
    print_log('{} | Epo: {:02d}/{:02d}, '
        'It: {:04d}/{:04d}, '
        'EP: {:s}, ETA: {:s}, seq {:s}, frame {:05d}, {}'
        .format(cfg, epoch, total_epoch, iter, total_iter, \
        convert_secs2time(ep), convert_secs2time(ep / iter * (total_iter * (total_epoch - epoch) - iter)), seq, frame, losses_str), log)


def get_timestring():
    return time.strftime('%Y%m%d_%Hh%Mm%Ss')


def batch_to_device(batch, device):
    for key in batch:
        for i in range(len(batch[key])):
            batch[key][i] = batch[key][i].to(device)
            
    return batch