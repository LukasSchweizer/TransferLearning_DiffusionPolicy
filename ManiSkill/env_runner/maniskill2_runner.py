import wandb
import numpy as np
import torch
import tqdm
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

# Maniskill dependencies
import gymnasium as gym
import collections
import mani_skill2.utils.sapien_utils as utils
import mani_skill2.envs


class ManiSkill2Runner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=10,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        # def env_fn():
        #     return MultiStepWrapper(
        #         SimpleVideoRecordingWrapper(
        #             MujocoPointcloudWrapperAdroit(env=AdroitEnv(env_name=task_name, use_point_cloud=True),
        #                                           env_name='adroit_'+task_name, use_point_crop=use_point_crop)),
        #         n_obs_steps=n_obs_steps,
        #         n_action_steps=n_action_steps,
        #         max_episode_steps=max_steps,
        #         reward_agg_method='sum',
        #     )
        cprint(self.task_name, 'red')
        self.eval_episodes = eval_episodes
        self.env = gym.make(
            self.task_name,
            obs_mode="pointcloud",
            control_mode="pd_joint_pos", 
            render_mode="human"
            )

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        rewards = []

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Maniskill2 {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
            obs, _ = env.reset(seed=0, options=dict(model_id="5000"))
            policy.reset()

            # keep a queue of last 2 steps of observations
            obs_deque = collections.deque(
                [obs] * self.n_obs_steps, maxlen=self.n_obs_steps)

            step_idx = 0
            done = False
            num_goal_achieved = 0
            actual_step_count = 0
            while not done:
                # create obs dict
                pointcloud = np.stack([x["pointcloud"]["xyzw"] for x in obs_deque])
                agent_poses = np.stack([np.concatenate((x["agent"]["qpos"], x["agent"]["qvel"])).flatten() for x in obs_deque])
                data = {
                    "point_cloud": pointcloud,
                    "agent_pos": agent_poses,
                }
                
                np_obs_dict = dict(data)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))
                
                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                # step env
                for i in range(len(action)):
                    obs, reward, done, _, info = env.step(action[i])
                    # all_goal_achieved.append(info['goal_achieved']
                    # print(f"Step {actual_step_count} Reward: {reward} Done: {done} Success: {info['success']}")
                    obs_deque.append(obs)
                    done = np.all(done)
                    actual_step_count += 1
                    if actual_step_count >= self.max_steps or info['success']:
                        done = True
                        break
                    env.render()
            
            rewards.append(round(reward, 4))
            num_goal_achieved += np.sum(info['success'])
            all_success_rates.append(info['success'])
            all_goal_achieved.append(num_goal_achieved)

        # log
        log_data = dict()
        

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(rewards)
        cprint(f"test_mean_score: {np.mean(rewards)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        # videos = env.env.get_video()
        # if len(videos.shape) == 5:
        #     videos = videos[:, 0]  # select first frame
        # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        # log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None
        env.close()

        return log_data
