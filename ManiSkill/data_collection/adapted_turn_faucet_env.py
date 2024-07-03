from typing import Dict, List, Union

import numpy as np
import sapien.physx as physx
import torch

from mani_skill.envs.tasks.tabletop.turn_faucet import TurnFaucetEnv
from mani_skill.utils.registration import register_env
from mani_skill.envs.utils import randomization
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose

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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            b = len(env_idx)
            p = torch.zeros((b, 3))
            p[:, :2] = randomization.uniform(-0.05, 0.05, size=(b, 2))
            p[:, 2] = self.model_offsets[:, 2]
            # p[:, 2] = 0.5
            # ori = self._episode_rng.uniform(-np.pi / 12, np.pi / 12)
            q = randomization.random_quaternions(
                n=b, lock_x=True, lock_y=True, bounds=(-torch.pi / 12, torch.pi / 12)
            )
            self.faucet.set_pose(Pose.create_from_pq(p, q))

            # apply pose changes and update kinematics to get updated link poses.
            if physx.is_gpu_enabled():
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            cmass_pose = (
                    self.target_switch_link.pose * self.target_switch_link.cmass_local_pose
            )
            self.target_link_pos = cmass_pose.p
            joint_pose = (
                self.target_switch_link.joint.get_global_pose().to_transformation_matrix()
            )
            self.target_joint_axis[env_idx] = joint_pose[env_idx, :3, 0]
            # self.handle_link_goal.set_pose(cmass_pose)

