import argparse
import gymnasium as gym
from data_collection.debug import plot_rgb


def register_adapted_envs():
    # Register AdaptedTurnFaucetEnv
    gym.envs.registration.register(
        id='AdaptedTurnFaucet-v0',
        entry_point='data_collection.adapted_turn_faucet_env:AdaptedTurnFaucetEnv',
        max_episode_steps=500,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="AdaptedTurnFaucet-v0")
    parser.add_argument("-o", "--obs-mode", type=str, default="state")
    parser.add_argument("-r", "--robot-uid", type=str, default="panda", help="Robot setups supported are ['panda']")
    parser.add_argument("--object-id", type=str, default=None)
    args, opts = parser.parse_known_args()

    return args


# Register Adapted Envs
register_adapted_envs()
# Parse command line arguments
args = parse_args()
# Create gym environment
env = gym.make(
    args.env_id,
    obs_mode="image", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="human",
    asset_root="../data/partnet_mobility/dataset",
)
# Reset environment & run it
obs, _ = env.reset(seed=0)  # reset with a seed for determinism

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    plot_rgb(obs, "top_down_camera")
    quit()
    done = terminated
    env.render()  # a display is required to render
env.close()
