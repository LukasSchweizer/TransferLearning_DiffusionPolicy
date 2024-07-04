from __future__ import annotations
import zarr
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer


class RobotReplayBuffer(ReplayBuffer):
    def __init__(self, root: zarr.Group):
        super().__init__(root)
        return

    def add_episode_from_list(self, data_list: list[dict[str, np.ndarray]], **kwargs):
        """
        data_list is a list of dictionaries, where each dictionary contains the data for one step.
        """
        data_dict = dict()
        for key in data_list[0].keys():
            data_dict[key] = np.stack([x[key] for x in data_list])
        self.add_episode(data_dict, **kwargs)
        return
