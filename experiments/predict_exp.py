"""
Created on Feb 7

@author: lachaji

"""

import os, sys
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT_DIRECTORY = './../'
ROOT_DIRECTORY = os.path.abspath( os.path.join(os.getcwd(), ROOT_DIRECTORY) )

sys.path.append(ROOT_DIRECTORY)
os.chdir(ROOT_DIRECTORY)


from utils.SynCanDataset import SynCanDataset
from torch.utils.data import DataLoader

from utils.timer import Timer
from utils.utils import seed_all, load_model_best, print_log, get_timestring
from utils.config import Config

from utils.predictor import run_statistics, run_test
from utils.optimizer import SimpleLossCompute
from utils.loss import L2Loss
from models.model import MultiIDModel

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_curve, recall_score

timer = Timer()

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='multiagent_bert_test')
parser.add_argument('--tmp', action='store_true', default=False)
parser.add_argument('--val_test', type=int, default=1,
                    help='0 if Validation, 1 if Test')
parser.add_argument('--test_file', default='test_suppress')

args = parser.parse_args()

cfg = Config(args.cfg, args.tmp, create_dirs=True)

seed_all(cfg.seed)
torch.set_default_dtype(torch.float32)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

time_str = get_timestring()
log = open(os.path.join(cfg.log_dir, 'log_predict.txt'), 'a+')

print_log("time str: {}".format(time_str), log)



print_log('\n',log)
print_log(" Instanciating model ".center(70, "="), log)


### Loading pretrained model
if(args.val_test==1 and (args.test_file == 'test_suppress' or args.test_file=='flooding')):
    model_ctx_update = {}
    model_ctx_update['input_max_len'] = [5000] * 10
else:
    model_ctx_update = None

model, state = load_model_best(MultiIDModel, cfg.load_from, device, model_ctx_update)

n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print_log(f" Number of learnable params {n_total_params}".ljust(30), log)


model_ctx = state['model_params']
optim_ctx = cfg.optim_params

data_loader_ctx = cfg.data_loader
data_loader_ctx['data_path'] = cfg.data_path
data_loader_ctx['backprop_window'] = model_ctx['backprop_window']


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

valdata = SynCanDataset(data_loader_ctx, 'val')
val_loader = DataLoader(
        valdata,
        batch_size = BATCH_SIZE,
        num_workers=cfg.data_loader.num_workers,
        shuffle = False
)

print_log(f"Nb samples in validation set {''.ljust(8)} : {len(valdata)}", log)  

if(args.val_test == 1):
    testdata = SynCanDataset(data_loader_ctx, args.test_file)
    test_loader = DataLoader(
            testdata,
            batch_size = BATCH_SIZE,
            num_workers=cfg.data_loader.num_workers,
            shuffle = False
    )
    print_log(f"Nb samples in test set {''.ljust(14)} : {len(testdata)}", log)  
 

print_log(f"Time spend : {timer.tak()}", log)



### Loss
if optim_ctx['loss']=="l2loss":
    criterion = L2Loss(per_sequence_loss=True)
else:
    print_log(f"the loss <{optim_ctx['loss']}> is not known", log)
    exit()
    

loss_compute_val = SimpleLossCompute(criterion, None)



#%%

if(optim_ctx['threshold'] == - 1):
    print_log('\n',log)
    print_log(" Start loss statistics calculation ".center(70, "="), log)
    
    
    losses = run_statistics(model, log, val_loader, data_loader_ctx, loss_compute_val, device, verbose=cfg.print_freq)
    
    print_log(f" Loss Max Value : {losses.max()}", log)
    print_log(f" Loss Min Value : {losses.min()}", log)
    print_log(f" Loss Mean Value : {losses.mean()}", log)
    print_log(f" Loss Var Value : {losses.std()}", log)

    print_log(f" Per Loss > 0.01 : {len(losses[losses > 0.01]) / len(losses)}", log)
    print_log(f" Per Loss > 0.1 : {len(losses[losses > 0.1]) / len(losses)}", log)

    plt.figure()
    plt.hist(losses, range=(losses.min(), losses.max()))
    plt.show()
    plt.savefig(os.path.join(cfg.result_dir, 'normal_loss_histogram_minmax.png'))

    plt.figure()
    plt.hist(losses, range=(losses.min(), 0.01))
    plt.show()
    plt.savefig(os.path.join(cfg.result_dir, 'normal_loss_histogram_001.png')) 

    plt.figure()
    plt.hist(losses, range=(losses.min(), 0.001))
    plt.show()
    plt.savefig(os.path.join(cfg.result_dir, 'normal_loss_histogram_0001.png')) 
    
else:
    threshold = optim_ctx['threshold']
    
    print_log(f"The validation threshold is {threshold}", log)
    
    if(args.val_test == 0):
        print_log(" Start Testing On Normal data ".center(70, "="), log)

        p_labels, gt_labels = run_test(model, log, val_loader, data_loader_ctx, loss_compute_val, threshold, device, verbose=50) 
        acc = accuracy_score(gt_labels, p_labels)     
        print_log(f" Accuracy on normal data : {acc*100} %", log)
    else:
        print_log(f" Start Testing On {args.test_file} data ".center(70, "="), log)
        p_labels, gt_labels = run_test(model, log, test_loader, data_loader_ctx, loss_compute_val, threshold, device, verbose=50) 
        acc = accuracy_score(gt_labels, p_labels)
        f1 = f1_score(gt_labels, p_labels)
        roc = roc_curve(gt_labels, p_labels)
        recall = recall_score(gt_labels, p_labels)
        print_log(f" Accuracy on {args.test_file} data : {acc*100} %", log)
        print_log(f" F1 on {args.test_file} data : {f1}", log)
        print_log(f" Recall on {args.test_file} data : {recall}", log)
        print_log(f" ROC on {args.test_file} data : {roc}", log)
        
    
        
    
    
    
    
    