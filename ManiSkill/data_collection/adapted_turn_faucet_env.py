import numpy as np
from mani_skill2.envs.misc.turn_faucet import TurnFaucetEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import (
    look_at,
)


@register_env("AdaptedTurnFaucet-v0", max_episode_steps=500, override=True)
class AdaptedTurnFaucetEnv(TurnFaucetEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _register_cameras(self):
        cameras = []
        # Add base camera
        base_camera_pose = look_at([-0.4, 0, 0.3], [0, 0, 0.1])
        base_camera = CameraConfig("base_camera", base_camera_pose.p, base_camera_pose.q, 128, 128,
                                   np.pi / 2, 0.01, 10)
        cameras.append(base_camera)
        # Add top down camera
        top_down_camera_pose = look_at([0, 0, 1.0], [0, 0, 0])
        top_down_camera = CameraConfig("top_down_camera", top_down_camera_pose.p, top_down_camera_pose.q, 128,
                                       128, np.pi / 2, 0.01, 10)
        cameras.append(top_down_camera)

        return cameras
