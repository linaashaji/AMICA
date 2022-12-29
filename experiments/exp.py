"""
Created on Feb 7

@author: lachaji

"""

import os, sys
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

ROOT_DIRECTORY = './../'
ROOT_DIRECTORY = os.path.abspath( os.path.join(os.getcwd(), ROOT_DIRECTORY) )

sys.path.append(ROOT_DIRECTORY)
os.chdir(ROOT_DIRECTORY)


from utils.SynCanDataset import SynCanDataset
from torch.utils.data import DataLoader

from utils.timer import Timer
from utils.utils import seed_all, load_state_best, print_log, get_timestring
from utils.config import Config

from utils.trainer import train, run_epoch
from utils.optimizer import NoamOpt, StepOpt, SimpleLossCompute
from utils.loss import L2Loss
from models.model import MultiIDModel


timer = Timer()

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='multiagent_bert')
parser.add_argument('--tmp', action='store_true', default=False)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--train_test', type=int, default=0,
                    help='0 if Train, 1 if Test')
args = parser.parse_args()

cfg = Config(args.cfg, args.tmp, create_dirs=True)

seed_all(cfg.seed)
torch.set_default_dtype(torch.float32)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_loader_ctx = cfg.data_loader
data_loader_ctx['data_path'] = cfg.data_path


model_ctx = cfg.transformer
optim_ctx = cfg.optim_params

data_loader_ctx['backprop_window'] = model_ctx['backprop_window']

time_str = get_timestring()
log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')

print_log("time str: {}".format(time_str), log)
writer = SummaryWriter(cfg.tb_dir)
tb_ind = 0

print_log(f" \n Model name: {cfg.model_name}", log)
print_log(" \n DataLoader parameters", log)
for k, v in data_loader_ctx.items():
    print_log(f" - {k.ljust(25)} {v}", log) 

print_log(" \n Model parameters", log)
for k, v in model_ctx.items():
    print_log(f" - {k.ljust(25)} {v}", log)
    
print_log(" \n Optimization parameters", log)
for k, v in optim_ctx.items():
    print_log(f" - {k.ljust(25)} {v}", log) 

BATCH_SIZE = cfg.data_loader.batch_size

### Load Dataset
timer.tik()

print_log('\n',log)
print_log(" Loading Dataset ".center(70, "="), log)


#Data Loader 

traindata = SynCanDataset(data_loader_ctx, 'train')
train_loader = DataLoader(
        traindata,
        batch_size = BATCH_SIZE,
        num_workers=cfg.data_loader.num_workers,
        shuffle = True
)


valdata = SynCanDataset(data_loader_ctx, 'val')
val_loader = DataLoader(
        valdata,
        batch_size = BATCH_SIZE,
        num_workers=cfg.data_loader.num_workers,
        shuffle = False
)


print_log(f"Nb samples in training set {''.ljust(10)} : {len(traindata)}", log)  
print_log(f"Nb samples in validation set {''.ljust(8)} : {len(valdata)}", log)   

print_log(f"Time spend : {timer.tak()}", log)



print_log('\n',log)
print_log(" Instanciating model ".center(70, "="), log)

model = MultiIDModel(model_ctx).to(device)

# Xavier initialisation
if model_ctx['xavier_int']:
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print_log(f" Number of learnable params {n_total_params}".ljust(30), log)

### Loss
if optim_ctx['loss']=="l2loss":
    criterion = L2Loss()
else:
    print_log(f"the loss <{optim_ctx['loss']}> is not known", log)
    exit()
    
    
if optim_ctx['optim']=="adam":
    model_opt = torch.optim.Adam(model.parameters(), lr=float(optim_ctx["initial_lr"]))

elif optim_ctx['optim']=="noam":
    model_opt = NoamOpt( model_ctx['d_model'] , 2, optim_ctx['warmup_step'],
                    torch.optim.Adam( model.parameters(), lr=0, betas=(0.9, 0.98), 
                                    eps=1e-9
                                    ))
elif optim_ctx['optim']=="step":
    model_opt = StepOpt( torch.optim.Adam( model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
                            initial_lr = float(optim_ctx["initial_lr"]),
                            step_size = optim_ctx['scheduler_step_size'] * len(traindata) / BATCH_SIZE,
                            warmup=optim_ctx['warmup_step'] * len(traindata) / BATCH_SIZE
                                     )
else:
    print_log(f"the optimizer <{optim_ctx['optim']}> is not known", log)
    exit()

loss_compute_train = SimpleLossCompute(criterion, model_opt)
loss_compute_val = SimpleLossCompute(criterion, None)

### Loading model if it already exist

state = load_state_best(model, model_opt, cfg.model_path, device)

#%%
epoch_start = 0
training_info = {"training_scores" : {"train" : [], "val" : [], "test":[]}}

if state is not None:
    training_info = state.get('training_info', training_info)
    epoch_start = training_info.get('epoch', 0)
    model_opt._step = training_info.get('steps', 1)


if(device.type == 'cpu'):
    verbose = 10
else:
    verbose = cfg.print_freq
    
if(args.train_test == 0):
        train(model, model_ctx, data_loader_ctx, training_info, cfg, log, model_opt,
              loss_compute_train, loss_compute_val, train_loader, val_loader, None,
              epoch_start, cfg.num_epochs, writer, device, verbose=verbose,
              run_epoch_func=run_epoch, metrics = ['Loss'])
