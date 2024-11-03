#-*- coding:utf-8 -*-
import math
import copy
import torch
import logging

log_mapping = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'critical': logging.CRITICAL,
}

from torch import nn
from inspect import isfunction
from functools import partial
from torch.utils import data
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import warnings
from skimage.util import montage as montage2d
warnings.filterwarnings("ignore", category=UserWarning)

torch.manual_seed(1234)

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

try:
    from apex import amp
    APEX_AVAILABLE = True
    print("APEX: ON")
except:
    APEX_AVAILABLE = False
    print("APEX: OFF")




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


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        opt,
        denoise_fn,
    ):
        super().__init__()
        self.channels = opt['denoising_model']['output_channels']
        self.image_size = opt['denoising_model']['input_size']
        self.depth_size = opt['denoising_model']['depth_size']
        self.denoise_fn = denoise_fn
        self.loss_type = opt['denoising_model']['loss_type']
    

        if opt['denoising_model']['loss_type'] == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif opt['denoising_model']['loss_type'] == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')

        self.set_new_noise_schedule(opt, opt['diffusion_parameters']['timesteps'])

    def set_new_noise_schedule(self, opt, timesteps):
        betas = make_beta_schedule(
            schedule=opt['diffusion_parameters']['scheduler'],
            n_timestep=timesteps,
            linear_start=opt['diffusion_parameters']['linear_start'],
            linear_end=opt['diffusion_parameters']['linear_end']
        )

        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))        

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        to_torch = partial(torch.tensor, dtype=torch.float32, device = 'cuda')

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t,  distance_tuple, contrast_scenario, clip_denoised: bool, c = None):

        bsize = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(bsize).to(x.device)
    
        left_volume = c[:,0:1]
        right_volume = c[:,1:2]

        x_recon = self.predict_start_from_noise(
            x,
            t = t,
            noise = self.denoise_fn(
                                        x = torch.cat([left_volume, x, right_volume], 1), 
                                        local_context = distance_tuple,
                                        global_context = contrast_scenario,
                                        timesteps = noise_level
                                    )
            )
                   
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance


    @torch.no_grad()
    def p_sample(self, x, t, condition_tensors, distance_tuple, contrast_scenario, clip_denoised=True):

        model_mean, model_log_variance = self.p_mean_variance(
                                                                x=x, 
                                                                t=t, 
                                                                distance_tuple = distance_tuple,
                                                                contrast_scenario = contrast_scenario,
                                                                clip_denoised=clip_denoised, 
                                                                c=condition_tensors
                                                                )
        
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()



    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors, distance_tuple, contrast_scenario):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):       
 
            img = self.p_sample(
                                    img, 
                                    i, 
                                    condition_tensors = condition_tensors, 
                                    distance_tuple = distance_tuple,
                                    contrast_scenario = contrast_scenario
                                )
        return img



    @torch.no_grad()
    def sample(self, batch_size, condition_tensors, distance_tuple, contrast_scenario):

        image_size = self.image_size
        depth_size = self.depth_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, depth_size, image_size, image_size), 
                                  condition_tensors = condition_tensors, 
                                  distance_tuple = distance_tuple,
                                  contrast_scenario = contrast_scenario)

    
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gamma
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )


    def p_losses(self, x_start, condition_tensors, distance_tuple, contrast_scenario, noise = None):

        b, c, h, w, d = x_start.shape

        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1, 1), noise=noise)

        assert condition_tensors.shape[1] == 2
        left_volume = condition_tensors[:,0:1]
        right_volume = condition_tensors[:,1:2]

        x_recon = self.denoise_fn(
                                    x = torch.cat([left_volume, x_noisy, right_volume], 1), 
                                    local_context = distance_tuple,
                                    global_context = contrast_scenario,
                                    timesteps = continuous_sqrt_alpha_cumprod
                                )

        loss = self.loss_func(noise, x_recon)

        return loss
    

    def forward(self, x, condition_tensors, distance_tuple, contrast_scenario, *args, **kwargs):

        b, c, d, h, w, img_size, depth_size = *x.shape, self.image_size, self.depth_size
          
        assert h == img_size and w == img_size and d == depth_size, f'Expected dimensions: height={img_size}, width={img_size}, depth={depth_size}. Actual: height={h}, width={w}, depth={d}.'
        return self.p_losses(
                                x, 
                                condition_tensors=condition_tensors, 
                                distance_tuple = distance_tuple,
                                contrast_scenario = contrast_scenario,
                                *args, **kwargs
                            )

class Runner(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        train_dataset,  
        val_dataset
        ):
        super().__init__()

        
        logging.basicConfig(level=log_mapping[opt['loglevel']])
        assert torch.cuda.is_available()
        self.experiment_output_path = opt['experiment_output_path']
    
        self.model = diffusion_model

        #ema init
        self.ema = EMA(opt['training']['ema']['ema_decay'])
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = opt['training']['ema']['update_ema_every']
        self.step_start_ema = opt['training']['ema']['step_start_ema']
        
        self.ds = train_dataset
        self.val_ds = val_dataset

        self.train_batch_size = opt['training']['batch_size']
        self.train_iterations = opt['training']['train_iterations']
        self.train_lr = opt['training']['train_lr']
        self.lr_decay_factor = opt['training']['lr_decay_factor']
        self.lr_decay_steps = [(self.train_iterations//3), (self.train_iterations//3)*2] # Decay at 1/3 and 2/3 progress of training.

        self.opt = Adam(self.model.parameters(), lr=self.train_lr)
        self.dl = cycle(data.DataLoader(self.ds, batch_size = self.train_batch_size, shuffle=True, num_workers=opt['training']['num_workers'], pin_memory=True))


        self.fp16 = opt['fp16'] 
        assert not self.fp16 or self.fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'
        
        if self.fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')
        
        self.model = nn.DataParallel(self.model)
        self.ema_model = nn.DataParallel(self.ema_model)


        self.log_dir = os.path.join(self.experiment_output_path, 'logs')
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.reset_parameters()

        self.save_and_sample_every = opt['training']['save_and_sample_every']
        self.print_loss_every = opt['training']['print_loss_every']
        self.step = 0

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
            'lr': self.train_lr
        }
        
        torch.save(data, os.path.join(self.experiment_output_path,f'model-{milestone}.pt'))


    def load(self, path):
        logging.info("Specified weight path: {}".format(path))
        data = torch.load(path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.train_lr = data['lr']

        for param_group in self.opt.param_groups:
            param_group['lr'] = self.train_lr


        logging.info('Weight loaded!')

        # TODO remove 
        self.step = 0
        

    def train(self):
        
        self.model.train()
        self.ema_model.train()
        logging.info('training start')

        while self.step < self.train_iterations:
            data = next(self.dl)

            input_tensors = data['input'].cuda()  # left and right scans
            target_tensors = data['target'].cuda()  # ground truth target scan
            
            distance_tuple = data['distance_tuple'].cuda() # local context
            contrast_scenario = data['contrast_scenario'].cuda() # global context

            for param in self.model.parameters():
                param.grad = None

            loss = self.model(target_tensors, 
                                condition_tensors=input_tensors,
                                distance_tuple = distance_tuple,
                                contrast_scenario = contrast_scenario
                            )
            b, c, d, h, w = target_tensors.shape
            loss = loss.sum()/int(b*c*d*h*w)
            
            if self.fp16:
                with amp.scale_loss(loss, self.opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.opt.step()

            if self.step % self.print_loss_every == 0:
                loss_ = loss.item()
                message = 'step: {}, training loss: {:3f}'.format(self.step, loss_)
                logging.info(message)
                self.writer.add_scalar("training_loss", loss_, self.step)


            if self.step % self.update_ema_every == 0:
                self.step_ema()


            if self.step in self.lr_decay_steps:
                for param_group in self.opt.param_groups:
                    print('old lr: ',  param_group['lr'])
                    param_group['lr'] *= self.lr_decay_factor 
                    self.train_lr = param_group['lr']
                    print('new lr: ',  param_group['lr'])


            # intermediate sampling
            if (self.step != 0 and self.step % self.save_and_sample_every == 0) or (self.step == self.train_iterations-1):
                milestone = self.step // self.save_and_sample_every

                sample_batch_size = 1 #self.batch_size
                sample_data = self.val_ds.random_sample_data(sample_batch_size)

                sample_data_condition_volumes = torch.cat([d['input'][np.newaxis] for d in sample_data], 0).cuda()
                sample_data_distance_tuple = torch.cat([d['distance_tuple'][np.newaxis] for d in sample_data], 0).cuda()
                sample_data_contrast_scenario = torch.cat([d['contrast_scenario'][np.newaxis] for d in sample_data], 0).cuda()

                sample_data_gt_volume = torch.cat([d['target'][np.newaxis] for d in sample_data], 0)

                names = [d['metadata']['name'] for d in sample_data]
                idxs = [d['metadata']['idx'] for d in sample_data]

                self.ema_model.eval()
                with torch.no_grad():
                    sampled_prediction = self.model.module.sample(
                                                                batch_size=sample_batch_size, 
                                                                condition_tensors = sample_data_condition_volumes, 
                                                                distance_tuple = sample_data_distance_tuple,
                                                                contrast_scenario = sample_data_contrast_scenario
                                                            )

                sampled_predicted_target_np = ((sampled_prediction.cpu().numpy() + 1.0) * 127.5).round().astype(np.uint8)
                sample_data_gt_volume_np = ((sample_data_gt_volume.numpy() + 1.0) * 127.5).round().astype(np.uint8)
                sample_condition_volume_np = ((sample_data_condition_volumes.cpu().numpy() + 1.0) * 127.5).round().astype(np.uint8)

                for b in range(sample_batch_size):
                    generated = sampled_predicted_target_np[b][0]
            
                    gt = sample_data_gt_volume_np[b][0]
                    left = sample_condition_volume_np[b][0]
                    right = sample_condition_volume_np[b][1]
                    
                    self.writer.add_image('intermediate_sampled_{}_{}_{}'.format(milestone, names[b], idxs[b]),
                                          np.hstack([montage2d(left), montage2d(generated), montage2d(right)]),
                                          b, dataformats='HW') 
                    
                    self.writer.add_image('intermediate_gt_{}_{}_{}'.format(milestone, names[b], idxs[b]),
                                          montage2d(gt),
                                          b,dataformats='HW') 

                self.ema_model.train()
                self.save(milestone)

            self.step += 1

        print('training completed')
        self.writer.close()