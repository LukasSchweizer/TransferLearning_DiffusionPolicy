from typing import Union

import zarr
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

from mani_skill2.utils import common
from mani_skill2.utils.io_utils import load_json
from utils.replay_buffer import RobotReplayBuffer
from utils.filter_pointcloud import filter_pointcloud_by_segmentation, downsample_point_clouds


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

# pulled from Maniskill3, used to index dict array
def index_dict_array(x1, idx: Union[int, slice], inplace=True):
    """Indexes every array in x1 with slice and returns result."""
    if (
        isinstance(x1, np.ndarray)
        or isinstance(x1, list)
        or isinstance(x1, torch.Tensor)
    ):
        return x1[idx]
    elif isinstance(x1, dict):
        if inplace:
            for k in x1.keys():
                x1[k] = index_dict_array(x1[k], idx, inplace=inplace)
            return x1
        else:
            out = dict()
            for k in x1.keys():
                out[k] = index_dict_array(x1[k], idx, inplace=inplace)
            return out


class ManiSkillTrajectoryDataset(Dataset):
    """
    A general torch Dataset you can drop in and use immediately with just about any trajectory .h5 data generated from ManiSkill.
    This class simply is a simple starter code to load trajectory data easily, but does not do any data transformation or anything
    advanced. We recommend you to copy this code directly and modify it for more advanced use cases

    Args:
        dataset_file (str): path to the .h5 file containing the data you want to load
        load_count (int): the number of trajectories from the dataset to load into memory. If -1, will load all into memory
        success_only (bool): whether to skip trajectories that are not successful in the end. Default is false
        device: The location to save data to. If None will store as numpy (the default), otherwise will move data to that device
        state_method (str): Which type of state observation to use. Full proprioception (the default) (qpos_qvel), position only (qpos), or tool control position (tcp)
    """

    def __init__(
        self, dataset_file: str, load_count=-1, success_only: bool = False, device=None, 
        zarr_path: str = None, state_method: str = "qpos_qvel",
    ) -> None:
        self.dataset_file = dataset_file
        self.device = device
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.replay_buffer = RobotReplayBuffer.create_from_path(zarr_path, mode="a")
        self.state_method = state_method

        self.obs = None
        self.actions = []
        self.terminated = []
        self.truncated = []
        self.success, self.fail, self.rewards = None, None, None
        if load_count == -1 or load_count >= len(self.episodes):
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            # if success_only:
            #     assert (
            #         "success" in eps
            #     ), "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
            #     if not eps["success"]:
            #         continue
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            eps_len = len(trajectory["actions"])

            # exclude the final observation as most learning workflows do not use it
            obs = index_dict_array(trajectory["obs"], slice(eps_len))

            self.actions.append(trajectory["actions"][1:])
            # self.terminated.append(trajectory["terminated"])
            # self.truncated.append(trajectory["truncated"])

            # handle data that might optionally be in the trajectory
            if "rewards" in trajectory:
                if self.rewards is None:
                    self.rewards = [trajectory["rewards"]]
                else:
                    self.rewards.append(trajectory["rewards"])
            if "success" in trajectory:
                if self.success is None:
                    self.success = [trajectory["success"]]
                else:
                    self.success.append(trajectory["success"])
            if "fail" in trajectory:
                if self.fail is None:
                    self.fail = [trajectory["fail"]]
                else:
                    self.fail.append(trajectory["fail"])
            self.generate_zarr(obs, trajectory["actions"])

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action = self.actions[idx]
        obs = common.index_dict_array(self.obs, idx, inplace=False)

        res = dict(
            obs=obs,
            action=action,
            terminated=self.terminated[idx],
            truncated=self.truncated[idx],
        )
        if self.rewards is not None:
            res.update(reward=self.rewards[idx])
        if self.success is not None:
            res.update(success=self.success[idx])
        if self.fail is not None:
            res.update(fail=self.fail[idx])
        return res

    def generate_zarr(self, obs, actions):
        data_dict = list()
        for i in range(len(actions)):
            pointcloud = obs["pointcloud"]["xyzw"][i].astype(np.float32)
            rgb = obs["pointcloud"]["rgb"][i].astype(np.float32)
            segmentation = obs["pointcloud"]["Segmentation"][i].astype(np.int8)
            # Remove floor from data (ENSURE ID 14 is FLOOR), this can be done by calling env.get_actors()
            segmented_pointcloud = filter_pointcloud_by_segmentation(pointcloud, segmentation, [14])
            segmented_rgb = filter_pointcloud_by_segmentation(rgb, segmentation, [14])

            # Uniformly sample points from pointcloud to ensure equal size
            num_points = 1024
            segmented_pointcloud, segmented_rgb = downsample_point_clouds([segmented_pointcloud, segmented_rgb], num_points)
            
            if self.state_method == "qpos_qvel":
                state = np.concatenate((obs["agent"]["qpos"][i], obs["agent"]["qvel"][i])).astype(np.float32)
            elif self.state_method == "qpos":
                state = obs["agent"]["qpos"][i].astype(np.float32)
            elif self.state_method == "tcp":
                state = obs["extra"]["tcp_pose"][i].astype(np.float32)
            elif self.state_method == "qpos_tcp":
                state = np.concatenate((obs["agent"]["qpos"][i], obs["extra"]["tcp_pose"][i])).astype(np.float32)
            else:
                print("No state method specified, defaulting to full proprioception (qpos + qvel).")
                state = np.concatenate((obs["agent"]["qpos"][i], obs["agent"]["qvel"][i])).astype(np.float32)
            data_dict.append({
                "pointcloud": segmented_pointcloud,
                "rgb": segmented_rgb,
                "state": state,
                "action": actions[i].astype(np.float32),
            })
        self.replay_buffer.add_episode_from_list(data_dict, compressors="disk")
        

