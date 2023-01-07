import torch
import os

checkpoint_filepath = 'results/multiagent_bert/model_newexp_3/models'
file_name = 'model_80_best.pt'
state = torch.load(os.path.join(checkpoint_filepath, file_name), map_location='cuda')

torch.save({"model_params" : state['model_params'],
            "model_state_dict" : state['model_state_dict']
            }, file_name)