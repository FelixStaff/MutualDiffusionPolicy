
import torch
from torch import nn

import collections

from model.common.ConvBlocks import Conv1dBlock, ConditionalResidualBlock1D, Downsample1d, Upsample1d
from model.common.SinusoidalPosEmbedding import SinusoidalPosEmb
from model.diffusion.conditional_unet_1d import ConditionalUnet1D

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

import numpy as np
import os
import gdown
from env.envData import PushTStateDataset, normalize_data, unnormalize_data

from env.pushT import PushTEnv

from tqdm import tqdm

import matplotlib.pyplot as plt

if __name__ == "__main__":

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
    
    #@markdown ### **Inference**

    # limit enviornment interaction to 200 steps before termination
    max_steps = 300
    # the output of PushTEnv
    obs_dim = 5
    action_dim = 2
    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    env = PushTEnv()
    # use a seed >200 to avoid initial states seen in the training dataset
    env.seed(100000)

    # get first observation
    obs, info = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode='human')]
    rewards = list()
    done = False
    step_idx = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model from the previous snippet
    ema_noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

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
    # Load the model from the previous snippet
    ema_noise_pred_net.load_state_dict(torch.load('model/saves/noise_pred_net_final.pth'))
    # Ema
    ema = EMAModel(
        parameters=ema_noise_pred_net.parameters(),
        power=0.75)
    # get scheduler
    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)
            # normalize observation
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            print ("Input shape: ", nobs.shape)
            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)
                print ("Obs_cond shape: ", obs_cond.shape)
                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                print ("Noisy action shape: ", noisy_action.shape)
                naction = noisy_action
                print ("Naction shape: ", naction.shape)
                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    print ("Noise pred shape: ", noise_pred.shape)
                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='human'))

                # update progress bar
                step_idx += 1
                #pbar.update(.5)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    print('Score: ', max(rewards))

    # visualize
    from IPython.display import Video
    #vwrite('vis.mp4', imgs)
    Video('vis.mp4', embed=True, width=256, height=256)