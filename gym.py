import gymnasium as gym
import collections
import numpy as np
import torch
import diffusion_policy
import pickle
import gzip
import os
import argparse
import torch.nn as nn
import mani_skill2.utils.sapien_utils as utils
import mani_skill2.envs
from models.base_models.vision_encoder import get_resnet, replace_bn_with_gn

from models.base_models.ConditionalUnet1D import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from models.datasets.image_dataset import normalize_data, unnormalize_data
from mani_skill2.agents.robots.panda import Panda

# from mani_skill2.envs.tasks.tabletop import TurnFaucetEnv

# def register_adapted_envs():
#     # Register AdaptedTurnFaucetEnv
#     gym.envs.registration.register(
#         id='AdaptedTurnFaucet-v1',
#         entry_point='ManiSkill.data_collection.adapted_turn_faucet_env:AdaptedTurnFaucetEnv',
#         max_episode_steps=200,
#     )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="TurnFaucet-v0")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgbd")
    parser.add_argument("-r", "--robot-uid", type=str, default="panda", help="Robot setups supported are ['panda']")
    parser.add_argument("--object-id", type=str, default=None)
    args, opts = parser.parse_known_args()

    return args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Register Adapted Envs
# register_adapted_envs()
# Parse command line arguments
args = parse_args()
# Create gym environment
env = gym.make(
    args.env_id,
    # num_envs=1,
    obs_mode="rgbd",
    control_mode="pd_joint_pos", # there is also "pd_joint_delta_pos", ...
    render_mode="human"
)

ckpt_path = "models/checkpoints/ema_nets_2024-07-09_17-14-43.pth"

print("... read data")
path = "demos/rigid_body/TurnFaucet-v0"
data_file = os.path.join(path, 'stats.gzip')
f = gzip.open(data_file,'rb')
stats = pickle.load(f)

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

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

vision_encoder = get_resnet('resnet18')

# IMPORTANT!
# replace all BatchNorm with GroupNorm to work with EMA
# performance will tank if you forget to do this!
vision_encoder = replace_bn_with_gn(vision_encoder)

# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 18
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 8

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
})

state_dict = torch.load(ckpt_path)
ema_nets = nets
ema_nets.load_state_dict(state_dict)
ema_nets.to(device)

# Reset environment & run it
obs, _ = env.reset(seed=0, options=dict(model_id=args.object_id))  # reset with a seed for determinism

# keep a queue of last 2 steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
rewards = list()
done = False
step_idx = 0

print(obs['extra'].keys())
done = False
while not done:
    B = 1
    # stack the last obs_horizon number of observations

    images = np.stack([x["image"]["hand_camera"]["rgb"] for x in obs_deque])
    agent_poses = np.stack([np.concatenate((x["agent"]["qpos"], x["agent"]["qvel"])).flatten() for x in obs_deque])
    print(agent_poses)

    # normalize observation
    nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
    # images are already normalized to [0,1]
    nimages = images
    nimages = nimages.reshape(obs_horizon, 3, 128, 128)

    # device transfer
    nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32)
    # (2,3,96,96)
    nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
    # (2,2)

    # infer action
    with torch.no_grad():
        # get image features
        image_features = ema_nets['vision_encoder'](nimages)
        # (2,512)

        # concat with low-dim observations
        obs_features = torch.cat([image_features, nagent_poses], dim=-1)

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, pred_horizon, action_dim), device=device)
        naction = noisy_action

        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = ema_nets['noise_pred_net'](
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

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
        env.render()  # a display is required to render
env.close()