
import torch
from torch import nn

from model.common.ConvBlocks import Conv1dBlock, ConditionalResidualBlock1D, Downsample1d, Upsample1d
from model.common.SinusoidalPosEmbedding import SinusoidalPosEmb
from model.diffusion.conditional_unet_1d import ConditionalUnet1D
from model.mine.ConditionedMine import MINEConditionalNet
from model.mine.MineLoss import mine_loss, train_loss_mine

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

import numpy as np
import os
import gdown
from env.envData import PushTStateDataset

from tqdm import tqdm

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Dataset Loading
    # download demonstration data from Google Drive
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    if not os.path.isfile(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256
    )
    #@markdown ### **Network Demo**

    # observation and action dimensions corrsponding to
    # the output of PushTEnv
    obs_dim = 5
    action_dim = 2
    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    # create MINE object
    mine = MINEConditionalNet(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # example inputs
    noised_action = torch.randn((1, pred_horizon, action_dim))
    obs = torch.zeros((1, obs_horizon, obs_dim))
    diffusion_iter = torch.zeros((1,))

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    noise = noise_pred_net(
        sample=noised_action,
        timestep=diffusion_iter,
        global_cond=obs.flatten(start_dim=1))

    # illustration of removing noise
    # the actual noise removal is performed by NoiseScheduler
    # and is dependent on the diffusion noise schedule
    denoised_action = noised_action - noise

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # device transfer
    device = torch.device('cpu')
    _ = noise_pred_net.to(device)

    #@markdown ### **Training**
    #@markdown
    #@markdown Takes about an hour. If you don't want to wait, skip to the next cell
    #@markdown to load pre-trained weights

    num_epochs = 100

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)
    
    # The Mine optimizer with adam
    mine_optimizer = torch.optim.AdamW(
        params=mine.parameters(),
        lr=1e-4, weight_decay=1e-6)
    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )


    tglobal = tqdm(range(num_epochs), desc='Epoch')
    global_loss = list()
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        tepoch = tqdm(dataloader, desc='Batch')
        for nbatch in tepoch:
            # data normalized in dataset
            # device transfer
            nobs = nbatch['obs'].to(device)
            naction = nbatch['action'].to(device)
            B = nobs.shape[0]

            # observation as FiLM conditioning
            # (B, obs_horizon, obs_dim)
            obs_cond = nobs[:,:obs_horizon,:]
            # (B, obs_horizon * obs_dim)
            obs_cond = obs_cond.flatten(start_dim=1)

            # sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)

            # predict the noise residual
            noise_pred = noise_pred_net(
                noisy_actions, timesteps, global_cond=obs_cond)
            
            # We input the noisy action and the predicted noise to the MINE
            # to estimate the MI between the noisy action and the predicted noise

            # We input the noisy action and the predicted noise to the MINE
            m_loss = mine_loss(
                mine, 
                noisy_actions, 
                noise_pred, 
                timesteps, 
                global_cond=obs_cond
            )

            # L2 loss
            loss = nn.functional.mse_loss(noise_pred, noise) + 0.01 * m_loss

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(noise_pred_net.parameters())
            # [TRAIN MINE]
            # predict the noise residual
            noise_pred = noise_pred_net(
                noisy_actions, timesteps, global_cond=obs_cond)
            
            mine.zero_grad()
            # update the MINE
            m_train_loss = train_loss_mine(
                mine, 
                noisy_actions, 
                noise_pred, 
                timesteps, 
                global_cond=obs_cond
            )
            mine_optimizer.zero_grad()
            m_train_loss.backward()
            mine_optimizer.step()


            # logging the loss
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            global_loss.append(loss_cpu)
            tepoch.set_postfix(loss=loss_cpu)

            
        # logging the loss in the epoch progress bar
        tglobal.set_postfix(loss=np.mean(epoch_loss))
        
        # Save the model
        torch.save(noise_pred_net.state_dict(), 'model/saves/noise_pred_net.pth')
        torch.save(mine.state_dict(), 'model/saves/mine.pth')
    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net = noise_pred_net
    ema.copy_to(ema_noise_pred_net.parameters())