"""
Created on Feb 7

@author: lachaji

"""

import torch
from torch.optim import lr_scheduler

class SimpleLossCompute:
    def __init__(self, criterion, opt=None):
        """
        criterion : loss to optimize
        opt : the optimizer
        max_norm : maximum norm for gradient clipping
        """
        self.criterion = criterion
        try:
            self.criterion.smooth
            self.smooth = True
        except:
            self.smooth = False
        self.opt = opt
            
    def __call__(self, *args, **kwargs):
        losses = self.criterion(*args, **kwargs)
        loss = None

        if not isinstance(losses, dict):
            losses = {"loss" : losses}
        
        for loss_name in losses.keys():
            if loss is None :
                loss = losses[loss_name]
            else:
                loss = loss + losses[loss_name]
        
        if self.opt is not None:
            try:
                self.opt.optimizer.zero_grad()
            except:
                self.opt.zero_grad()
            loss.backward()
            


            self.opt.step()
            try:
                self.opt.optimizer.zero_grad()
            except:
                self.opt.zero_grad()
        
        for loss_name in losses.keys():
            losses[loss_name] = losses[loss_name]
        losses['total_loss'] = loss
        return losses







        
class StepOpt:
    def __init__(self, optimizer, initial_lr=1e-4, gamma=0.5, step_size=10, warmup=0, cur_step=0):
        self.step_size = step_size
        self.warmup = warmup
        self.initial_lr = initial_lr
        self.gamma = gamma
        self.optimizer = optimizer
        self._step = cur_step
        self._rate = self.get_lr()

    def step(self):
        self._step += 1

        self._rate = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._rate

        self.optimizer.step()
    
    def get_lr(self):
        if self._step < self.warmup:
            return self.initial_lr * self._step / self.warmup

        factor = self.gamma ** ( (self._step-self.warmup) // self.step_size )
        return factor * self.initial_lr

        
def get_step_opt(model, initial_lr=1e-4, step_size=10, gamma=0.1, current_step=0):
    return StepOpt( 
            torch.optim.Adam( model.parameters(), lr=initial_lr, betas=(0.9, 0.98), 
                                   eps=1e-9
                                   ), 
        initial_lr, gamma, step_size, current_step
    )


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            ( self.model_size ** (-0.5) * 
             min(step ** (-0.5) , step * self.warmup **(-1.5) ) )
    
def get_noam_opt(model, d_model=32):
    return NoamOpt( d_model, 2, 4000,
                   torch.optim.Adam( model.parameters(), lr=0, betas=(0.9, 0.98), 
                                   eps=1e-9
                                   )
                  )


def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None, decay_gamma=0.1):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=decay_gamma)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler