from typing import Dict, List

import numpy as np
import torch

from mani_skill.envs.tasks.tabletop.turn_faucet import TurnFaucetEnv
from mani_skill.sensors import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.envs.utils.observations import (
    sensor_data_to_pointcloud,
    sensor_data_to_rgbd,
)


@register_env("AdaptedTurnFaucet-v1", max_episode_steps=200)
class AdaptedTurnFaucetEnv(TurnFaucetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        # Use specific faucet if its wanted
        if options.get("object_id", None) is not None:
            model_ids = np.array([options.get("object_id")])
        else:
            rand_idx = self._episode_rng.permutation(np.arange(0, len(self.all_model_ids)))
            model_ids = self.all_model_ids[rand_idx]

        model_ids = np.concatenate(
            [model_ids] * np.ceil(self.num_envs / len(self.all_model_ids)).astype(int)
        )[: self.num_envs]
        switch_link_ids = self._episode_rng.randint(0, 2 ** 31, size=len(model_ids))

        self._faucets = []
        self._target_switch_links: List[Link] = []
        self.model_offsets = []
        for i, model_id in enumerate(model_ids):
            # partnet-mobility is a dataset source and the ids are the ones we sampled
            # we provide tools to easily create the articulation builder like so by querying
            # the dataset source and unique ID
            model_info = self.train_info[model_id]
            builder = articulations.get_articulation_builder(
                self.scene,
                f"partnet-mobility:{model_id}",
                urdf_config=dict(density=model_info.get("density", 8e3)),
            )
            builder.set_scene_idxs(scene_idxs=[i])
            faucet = builder.build(name=f"{model_id}-{i}")
            for joint in faucet.active_joints:
                joint.set_friction(1.0)
                joint.set_drive_properties(0, 10.0)
            self.model_offsets.append(model_info["offset"])
            self._faucets.append(faucet)

            switch_link_names = []
            for j, semantic in enumerate(model_info["semantics"]):
                if semantic[2] == "switch":
                    switch_link_names.append(semantic[0])
            # import ipdb;ipdb.set_trace()
            switch_link = faucet.links_map[
                switch_link_names[switch_link_ids[i] % len(switch_link_names)]
            ]
            self._target_switch_links.append(switch_link)
            switch_link.joint.set_friction(0.1)
            switch_link.joint.set_drive_properties(0.0, 2.0)

        self.faucet = Articulation.merge(self._faucets, name="faucet")
        self.target_switch_link = Link.merge(self._target_switch_links, name="switch")
        self.model_offsets = common.to_tensor(self.model_offsets, device=self.device)
        self.model_offsets[:, 2] += 0.01  # small clearance

        # self.handle_link_goal = actors.build_sphere(
        #     self.scene,
        #     radius=0.03,
        #     color=[0, 1, 0, 1],
        #     name="switch_link_goal",
        #     body_type="kinematic",
        #     add_collision=False,
        # )

        qlimits = self.target_switch_link.joint.get_limits()
        qmin, qmax = qlimits[:, 0], qlimits[:, 1]
        self.init_angle = qmin
        self.init_angle[torch.isinf(qmin)] = 0
        self.target_angle = qmin + (qmax - qmin) * 0.9
        self.target_angle[torch.isinf(qmax)] = torch.pi / 2
        # the angle to go
        self.target_angle_diff = self.target_angle - self.init_angle
        self.target_joint_axis = torch.zeros((self.num_envs, 3), device=self.device)

    @property
    def _default_sensor_configs(self):
        cameras = []
        # Base Camera
        pose = sapien_utils.look_at([-0.4, 0, 0.3], [0, 0, 0.1])
        cameras.append(CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2))
        # Static Camera (not needed, because "base_camera" is already static)
        # pose = sapien.Pose(p=[1, 0, 0])
        # cameras.append(CameraConfig("static_camera", pose=pose, width=640, height=480, fov=np.pi / 2))
        return cameras

    def get_obs(self, info: Dict = None):
        # Get obs with state_dict data
        obs = super().get_obs(info)
        # Get RGBD data from sensor
        obs_sensor = self._get_obs_with_sensor_data(info)
        sensor_rgbd = sensor_data_to_rgbd(obs_sensor, self._sensors, rgb=True, depth=True, segmentation=True)
        # Get Pointcloud data from sensor
        obs_sensor = self._get_obs_with_sensor_data(info)
        sensor_point_cloud = sensor_data_to_pointcloud(obs_sensor, self._sensors)
        # Add data to observation
        obs['sensor_param'] = sensor_rgbd['sensor_param']
        obs['sensor_data'] = sensor_rgbd['sensor_data']
        obs['pointcloud'] = sensor_point_cloud['pointcloud']

        return obs


