"""
Created on Feb 7

@author: lachaji

"""

import copy
import torch.nn as nn

def clones(module, N):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([ copy.deepcopy(module) for _ in range(N) ])

def clamp_max(x, max_value):
    cst = torch.masked_fill(max_value/x, x<max_value, 1).detach()
    return x * cst
    
class Lambda(nn.Module):
    "Transform a function in a pytorch module"
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)