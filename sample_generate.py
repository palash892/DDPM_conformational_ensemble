import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
import os
import random
from pathlib import Path

import torch 
from torch.utils import data
from torch import nn

import denoising_diffusion_1D.ddpm_1d
from denoising_diffusion_1D.ddpm_1d import Unet, GaussianDiffusion, Trainer, Dataset_traj, cycle, num_to_groups

torch_seed = 42
torch.manual_seed(torch_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(torch_seed)


random_seed = 42
random.seed(random_seed)

np_seed = 42
np.random.seed(np_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


system = 'moon_data'



model = Unet(
    dim = 32,                   
    dim_mults = (1, 2, 4, 8 ),   
    groups = 8 
).to(device) 
model.to(device)

op_num = 2

diffusion = GaussianDiffusion(
    model,                        
    timesteps = 1000,             
    loss_type = 'l2'              
).to(device)



trainer = Trainer(
    diffusion,                                   
    folder = 'moon',                        
    system = system,  
    train_batch_size = 128,                      
    train_lr = 1e-5,                             
    train_num_steps = 50001,                   
    gradient_accumulate_every = 1,               
    ema_decay = 0.995,                           
    op_number = op_num,
    fp16 = False,                                 
    device = device
)


batch_size = 1280  

num_sample = 10000 
model_id = 8
trainer.load(model_id) 

batches = num_to_groups(num_sample, batch_size)
all_ops_list = list(map(lambda n: trainer.ema_model.sample(trainer.op_number, batch_size=n), batches))


all_ops = torch.cat(all_ops_list, dim=0).cpu()
all_ops = trainer.rescale_sample_back(all_ops)
generate_folder = Path(f'./generate_sample')
generate_folder.mkdir(exist_ok = True, parents=True)
np.save(str(generate_folder/f'generate_samples_{num_sample}_model_id_{model_id}_{system}'), all_ops.numpy())
print(str(generate_folder /f'generate_samples_{num_sample}_model_id_{model_id}_{system}.npy'))

