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
from utils.filter_pointcloud import filter_pointcloud_by_segmentation, downsample_point_clouds
from utils.visualize_data import plot_point_cloud


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
                 num_sampled_pts=1024,
                 state_method="qpos_qvel",
                 render_pointcloud=False,
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
            render_mode="human",
            camera_cfgs={"add_segmentation": True, "use_stereo_depth": False},
            )

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.num_sampled_pts = num_sampled_pts
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        self.state_method = state_method
        self.render_pointcloud = render_pointcloud

    def segment_pointcloud(self, pointcloud, segmentation):
        segmented_pointcloud = []
        for i in range(self.n_obs_steps):
            # Remove floor from data (ENSURE ID 14 is FLOOR), this can be done by calling env.get_actors()
            filtered_pointcloud = filter_pointcloud_by_segmentation(pointcloud[i], segmentation[i], [14])
            sampled_pointcloud = downsample_point_clouds([filtered_pointcloud], self.num_sampled_pts)
            segmented_pointcloud.append(sampled_pointcloud)
        return np.squeeze(np.stack(segmented_pointcloud))

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        eval_seeds = [0, 1, 2] #np.linspace(1040, 1042, num=3*self.eval_episodes, dtype=int)
        models = ["5001", "5006"]
        success_log = {}

        cprint(f"evaluating {self.eval_episodes} episodes using seeds {eval_seeds}", 'cyan')

        for model in models:
            cprint(f"evaluating model number {model}...")
            seed_successes = {}
            for seed_idx in tqdm.tqdm(range(len(eval_seeds)), desc="Seeded Eval", leave=False, mininterval=self.tqdm_interval_sec):
                seed_successes[eval_seeds[seed_idx]] = []
                rewards = []
                for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Maniskill2 {self.task_name} Pointcloud Env",
                                            leave=False, mininterval=self.tqdm_interval_sec):
                    # start rollout
                    obs, _ = env.reset(seed=eval_seeds[seed_idx], options=dict(model_id=model))
                    policy.reset()

                    # keep a queue of last 2 steps of observations
                    obs_deque = collections.deque(
                        [obs] * self.n_obs_steps, maxlen=self.n_obs_steps)

                    step_idx = 0
                    done = False
                    num_goal_achieved = 0
                    actual_step_count = 0
                    plot_pt_cloud = None
                    reward_keeper = []
                    while not done:
                        # create obs dict
                        pointcloud = np.stack([x["pointcloud"]["xyzw"] for x in obs_deque])
                        segmentation = np.stack([x["pointcloud"]["Segmentation"] for x in obs_deque])
                        processed_pointcloud = self.segment_pointcloud(pointcloud, segmentation)
                        plot_pt_cloud = processed_pointcloud

                        # Select state method
                        if self.state_method == "qpos_qvel":
                            agent_poses = np.stack([np.concatenate((x["agent"]["qpos"], x["agent"]["qvel"])).flatten() for x in obs_deque])
                        elif self.state_method == "qpos":
                            agent_poses = np.stack([x["agent"]["qpos"] for x in obs_deque])
                        elif self.state_method == "tcp":
                            agent_poses = np.stack([x["extra"]["tcp_pose"] for x in obs_deque]) 
                        elif self.state_method == "qpos_tcp":
                            agent_poses = np.stack([np.concatenate((x["agent"]["qpos"], x["extra"]["tcp_pose"])).flatten() for x in obs_deque])
                        
                        data = {
                            "point_cloud": processed_pointcloud,
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
                            #env.render()
                            reward_keeper.append(round(reward, 4))
                    if self.render_pointcloud:
                        plot_point_cloud(plot_pt_cloud, 0)
                    
                    rewards.append(np.mean(reward_keeper))
                    num_goal_achieved += np.sum(info['success'])
                    all_success_rates.append(info['success'])
                    all_goal_achieved.append(num_goal_achieved)
                    seed_successes[eval_seeds[seed_idx]].append(bool(info['success']))
                    seed_successes[eval_seeds[seed_idx]].append(np.mean(rewards))
                
                run_avg = np.mean(seed_successes[eval_seeds[seed_idx]][0])
                cprint(f"(Model {model}) Seed no. {eval_seeds[seed_idx]} avg success rate: {run_avg}, avg reward: {np.mean(rewards)}", 'magenta')
                
            success_log[model] = seed_successes

        # log
        log_data = dict()
        
        log_data['seeded_runs'] = success_log
        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        cprint(f"average_reward: {np.mean(rewards)}, test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        # cprint(log_data, 'red')

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
