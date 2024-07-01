import gymnasium as gym

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils import sapien_utils
from mani_skill.envs.tasks.tabletop import turn_faucet

env = gym.make(
    "TurnFaucet-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="state_dict", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="human"
)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
print(obs['extra'].keys())
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated
    env.render()  # a display is required to render
env.close()