import argparse

import gymnasium as gym

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils import sapien_utils
from mani_skill.envs.tasks.tabletop import turn_faucet


def register_adapted_envs():
    # Register AdaptedTurnFaucetEnv
    gym.envs.registration.register(
        id='AdaptedTurnFaucet-v1',
        entry_point='data_collection.adapted_turn_faucet_env:AdaptedTurnFaucetEnv',
        max_episode_steps=200,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="AdaptedTurnFaucet-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="state")
    parser.add_argument("-r", "--robot-uid", type=str, default="panda", help="Robot setups supported are ['panda']")
    parser.add_argument("--object-id", type=str, default=None)
    parser.add_argument("--record-dir", type=str, default="../data")
    args, opts = parser.parse_known_args()

    return args


register_adapted_envs()
args = parse_args()
env = gym.make(
    args.env_id,
    num_envs=1,
    obs_mode="state_dict", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="human"
)

obs, _ = env.reset(seed=0, options=dict(object_id=args.object_id)) # reset with a seed for determinism
print(obs['extra'].keys())
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated
    env.render()  # a display is required to render
env.close()
