import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import sys
import math
import copy
from tqdm import tqdm

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils import data
from pathlib import Path
from torch.optim import Adam

from inspect import isfunction
from functools import partial
from einops import rearrange


try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False





SAVE_AND_SAMPLE_EVERY = 6250
UPDATE_EMA_EVERY = 10
PRINT_LOSS_EVERY = 200
SAVE_LOSS_EVERY = 10






#~~~~~~~~~~~~~~~~~~~~~~~~some helper function
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)






#---------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~some helper module
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g







#--------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Building block 
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.mlp(time_emb)[:, :, None]

        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) l -> qkv b heads c l', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c l -> b (heads c) l', heads=self.heads)
        return self.to_out(out)



class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)






#----------------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~model architecture
class Unet(nn.Module):
    def __init__(self, dim, out_dim = None, dim_mults=(1, 2, 4, 8), groups = 8):
        super().__init__()
        dims = [1, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.feature_dim = dim
        self.dim_mults = dim_mults
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = dim, groups = groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim = dim, groups = groups),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim, groups = groups)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = dim, groups = groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = dim, groups = groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim = dim, groups = groups),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, 1)
        self.final_conv = nn.Sequential(
            Block(dim*2, dim, groups = groups),
            nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        size_list = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            size_list.append(x.shape[-1])
            x = downsample(x)


        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x[:,:,:size_list.pop()], h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        if x.shape[-1] != h[-1].shape[-1]:
            x = x[:, :, :h[-1].shape[-1]]

        x = torch.cat((x, h.pop()), dim = 1)

        return self.final_conv(x[:,:,:size_list.pop()])






#----------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~beta schedule
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
    return np.clip(betas, a_min = 0, a_max = 0.999)





#----------------------------------------------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~noising process(Gaussian diffusion)
class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, timesteps=1000, loss_type='l1', betas = None):
        super().__init__()
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = linear_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        


        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True):
        b, _, l, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if (t > 0).any() else 0.
        # print(t, noise)
        
        denosied_x = model_mean + (0.5 * model_log_variance).exp() * noise
        return denosied_x

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device

        b = shape[0]
        state = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            state = self.p_sample(state, torch.full((b,), i, device=device, dtype=torch.long))
        return state

    @torch.no_grad()
    def sample(self, op_number, batch_size = 128):
        return self.p_sample_loop((batch_size, 1, op_number))


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        
        return x_noisy

    def p_losses(self, x_start, t, noise = None):
        b, c, l= x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.denoise_fn(x_noisy, t)


        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)





#--------------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~dataset
class Dataset_traj(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, folder, system):
        super().__init__()
        self.folder = folder
        #self.paths = [p for p in Path(f'{folder}').glob(f'**/{system}_traj*.npy')]
        self.data = np.load(f'{folder}/{system}.npy')
        #self.data = self.data[:128, :]
        #print(self.data.shape)
        self.max_data = np.max(self.data, axis = 0)
        self.min_data = np.min(self.data, axis = 0)


  def __len__(self):
        'Denotes the total number of samples'
        return np.shape(self.data)[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.data[index:index+1, :]
        x = 2*x/(self.max_data - self.min_data)
        x = x - 2*self.min_data/(self.max_data-self.min_data) -1
        #x = x[ np.newaxis, ...]
        return torch.from_numpy(x).float()







#-------------------------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Trainer module
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        system,
        *,
        system_for_sample = None,
        ema_decay = 0.995,
        op_number = 4,
        train_batch_size = 128,
        sample_batch_size = None,
        train_lr = 1e-5,
        train_num_steps = 200001,
        gradient_accumulate_every = 1,
        fp16 = False,
        step_start_ema = 2000,
        device = None
    ):
        super().__init__()

        feature_dim = diffusion_model.denoise_fn.feature_dim
        dim_mults = diffusion_model.denoise_fn.dim_mults
        MODEL_INFO = f'{feature_dim }-'
        for w in dim_mults :
            MODEL_INFO +=  f'{w}-'
        MODEL_INFO  += f'b{train_batch_size}'

        self.RESULTS_FOLDER = Path(f'./results/{system}/{MODEL_INFO}')
        self.RESULTS_FOLDER.mkdir(exist_ok = True, parents=True)
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.step_start_ema = step_start_ema

        self.batch_size = train_batch_size
        self.op_number =  op_number
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = Dataset_traj(folder, system )
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))

        self.sample_batch_size = train_batch_size
        if system_for_sample == None:
            self.dl_sample= self.dl
        else:
            self.ds_sample = Dataset_traj(folder, system_for_sample )
            if sample_batch_size == None:
                self.sample_batch_size = train_batch_size
            self.dl_sample = cycle(data.DataLoader(self.ds_sample, batch_size = sample_batch_size, shuffle=True, pin_memory=True))

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.reset_parameters()
        self.req_loss = []
        self.req_step = []
        self.device = device

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'data_range': [self.ds.min_data, self.ds.max_data]
        }
        torch.save(data, str(self.RESULTS_FOLDER /  f'model-{milestone}.pt'))


    def rescale_sample_back(self, sample):

        def scale_back(data, minimums, maximums):
            data = (data + 1)/2.0*(maximums - minimums)
            data += minimums
            return data

        max_data = self.ds.max_data
        min_data = self.ds.min_data

        sample = scale_back(sample, min_data, max_data)

        return sample

    def load(self, milestone):
        model_data = torch.load(str(self.RESULTS_FOLDER / f'model-{milestone}.pt'))

        self.step =  model_data['step']
        self.model.load_state_dict( model_data['model'])
        self.ema_model.load_state_dict( model_data['ema'])
        self.ds.min_data = model_data['data_range'][0]
        self.ds.max_data = model_data['data_range'][1]
        self.dl = cycle(data.DataLoader(self.ds, batch_size = self.batch_size, shuffle=True, pin_memory=True))

    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).to(self.device)
                loss = self.model(data)
                backwards(loss / self.gradient_accumulate_every, self.opt)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % UPDATE_EMA_EVERY == 0:
                self.step_ema()

            if self.step % PRINT_LOSS_EVERY ==0:
                print("step for training = ", f'{self.step}, traing loss =  {loss.item()}')

            if self.step % SAVE_LOSS_EVERY ==0:
                self.req_loss.append(loss.item())
                self.req_step.append(self.step)


            if self.step != 0 and self.step % SAVE_AND_SAMPLE_EVERY == 0:
                milestone = self.step // SAVE_AND_SAMPLE_EVERY
                batches = num_to_groups(self.sample_batch_size, self.batch_size)
                all_ops_list = list(map(lambda n: self.ema_model.sample(self.op_number, batch_size=n), batches))
                all_ops = torch.cat(all_ops_list, dim=0).cpu()
                all_ops = self.rescale_sample_back(all_ops)
                np.save( str(self.RESULTS_FOLDER /  f'sample-{milestone}'), all_ops.numpy())
                self.save(milestone)

            self.step += 1

        print('the ddpm training has been completed')
