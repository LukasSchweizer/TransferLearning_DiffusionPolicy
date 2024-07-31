import argparse
import gymnasium as gym
from ManiSkill.data_collection.debug import plot_rgb, plot_point_cloud
from utils.filter_pointcloud import filter_pointcloud_by_segmentation, downsample_point_clouds
import numpy as np
import mani_skill2.envs


def plot_pointcloud(obs):
    pointcloud = obs["pointcloud"]["xyzw"].astype(np.float32)
    rgb = obs["pointcloud"]["rgb"].astype(np.float32)
    segmentation = obs["pointcloud"]["Segmentation"].astype(np.int8)
    # Remove floor from data (ENSURE ID 14 is FLOOR), this can be done by calling env.get_actors()
    segmented_pointcloud = filter_pointcloud_by_segmentation(pointcloud, segmentation, [14])
    segmented_rgb = filter_pointcloud_by_segmentation(rgb, segmentation, [14])

    # Uniformly sample points from pointcloud to ensure equal size
    num_points = 1024
    segmented_pointcloud, segmented_rgb = downsample_point_clouds([segmented_pointcloud, segmented_rgb], num_points)
    plot_point_cloud(segmented_pointcloud)

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
    obs_mode="pointcloud", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="human",
    asset_root="./data/partnet_mobility/dataset",
    camera_cfgs={"add_segmentation": True, "use_stereo_depth": False},
)
# Reset environment & run it
obs, _ = env.reset(seed=0, options=dict(model_id=args.object_id))  # reset with a seed for determinism


done = False
i = True
while not done:
    if i:
        i = False
        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        plot_pointcloud(obs)
    # env.step(None)

    done = terminated
    env.render()  # a display is required to render
    # plot_rgb(obs_rgb)
env.close()
