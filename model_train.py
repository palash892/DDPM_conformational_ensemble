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

print("model training started")
trainer.train()

np.savetxt(f"training_loss_{system}.txt", np.array([trainer.req_step, trainer.req_loss]).T, delimiter = "\t", fmt = "%0.3e")
print("model training completed")
