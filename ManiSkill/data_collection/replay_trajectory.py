"""Replay the trajectory stored in HDF5.
The replayed trajectory can use different observation modes and control modes.
We support translating actions from certain controllers to a limited number of controllers.
The script is only tested for Panda, and may include some Panda-specific hardcode.
"""

import argparse
import multiprocessing as mp
import os
from copy import deepcopy
from typing import Union

import gymnasium as gym
import h5py
import numpy as np
import sapien.core as sapien
from tqdm.auto import tqdm
from transforms3d.quaternions import quat2axangle

import mani_skill2.envs
from mani_skill2.agents.base_controller import CombinedController
from mani_skill2.agents.controllers import *
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.trajectory.merge_trajectory import merge_h5
from mani_skill2.utils.common import clip_and_scale_action, inv_scale_action
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import get_entity_by_name
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.trajectory.replay_trajectory import from_pd_joint_pos, from_pd_joint_delta_pos


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-path", type=str, required=True)
    parser.add_argument("-o", "--obs-mode", type=str, help="target observation mode")
    parser.add_argument(
        "-c", "--target-control-mode", type=str, help="target control mode"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--save-traj", action="store_true", help="whether to save trajectories"
    )
    parser.add_argument(
        "--save-video", action="store_true", help="whether to save videos"
    )
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--max-retry", type=int, default=0)
    parser.add_argument(
        "--discard-timeout",
        action="store_true",
        help="whether to discard timeout episodes",
    )
    parser.add_argument(
        "--allow-failure", action="store_true", help="whether to allow failure episodes"
    )
    parser.add_argument("--vis", action="store_true")
    parser.add_argument(
        "--use-env-states",
        action="store_true",
        help="whether to replay by env states instead of actions",
    )
    parser.add_argument(
        "--bg-name",
        type=str,
        default=None,
        help="background scene to use",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="number of demonstrations to replay before exiting. By default will replay all demonstrations",
    )
    return parser.parse_args(args)


def _main(args, proc_id: int = 0, num_procs=1, pbar=None):
    pbar = tqdm(position=proc_id, leave=None, unit="step", dynamic_ncols=True)

    # Load HDF5 containing trajectories
    traj_path = args.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]

    # Create a twin env with the original kwargs
    if args.target_control_mode is not None:
        ori_env = gym.make(env_id, **ori_env_kwargs)
    else:
        ori_env = None

    # Create a main env for replay
    target_obs_mode = args.obs_mode
    target_control_mode = args.target_control_mode
    env_kwargs = ori_env_kwargs.copy()
    if target_obs_mode is not None:
        env_kwargs["obs_mode"] = target_obs_mode
    if target_control_mode is not None:
        env_kwargs["control_mode"] = target_control_mode
    env_kwargs["bg_name"] = args.bg_name
    env_kwargs[
        "render_mode"
    ] = "rgb_array"  # note this only affects the videos saved as RecordEpisode wrapper calls env.render

    # ADDS SEGMENTATION TO OBSERVATION
    # env_kwargs["enable_gt_seg"] = True  # (NOT WORKING)
    env_kwargs["camera_cfgs"] = {"add_segmentation": True, "use_stereo_depth": False}

    env = gym.make(env_id, **env_kwargs)
    if pbar is not None:
        pbar.set_postfix(
            {
                "control_mode": env_kwargs.get("control_mode"),
                "obs_mode": env_kwargs.get("obs_mode"),
            }
        )

    # Prepare for recording
    output_dir = os.path.dirname(traj_path)
    ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    suffix = "{}.{}".format(env.obs_mode, env.control_mode)
    new_traj_name = ori_traj_name + "." + suffix
    if num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    env = RecordEpisode(
        env,
        output_dir,
        save_on_reset=False,
        save_trajectory=args.save_traj,
        trajectory_name=new_traj_name,
        save_video=args.save_video,
    )

    if env.save_trajectory:
        output_h5_path = env._h5_file.filename
        assert not os.path.samefile(output_h5_path, traj_path)
    else:
        output_h5_path = None

    episodes = json_data["episodes"][: args.count]
    n_ep = len(episodes)
    inds = np.arange(n_ep)
    inds = np.array_split(inds, num_procs)[proc_id]

    # Replay
    for ind in inds:
        ep = episodes[ind]
        episode_id = ep["episode_id"]
        traj_id = f"traj_{episode_id}"
        if pbar is not None:
            pbar.set_description(f"Replaying {traj_id}")

        if traj_id not in ori_h5_file:
            tqdm.write(f"{traj_id} does not exist in {traj_path}")
            continue

        reset_kwargs = ep["reset_kwargs"].copy()
        if "seed" in reset_kwargs:
            assert reset_kwargs["seed"] == ep["episode_seed"]
        else:
            reset_kwargs["seed"] = ep["episode_seed"]
        seed = reset_kwargs.pop("seed")

        ori_control_mode = ep["control_mode"]

        for _ in range(args.max_retry + 1):
            env.reset(seed=seed, options=reset_kwargs)
            if ori_env is not None:
                ori_env.reset(seed=seed, options=reset_kwargs)

            if args.vis:
                env.render_human()

            # Original actions to replay
            ori_actions = ori_h5_file[traj_id]["actions"][:]

            # Original env states to replay
            if args.use_env_states:
                ori_env_states = ori_h5_file[traj_id]["env_states"][1:]

            info = {}

            # Without conversion between control modes
            if target_control_mode is None:
                n = len(ori_actions)
                if pbar is not None:
                    pbar.reset(total=n)
                for t, a in enumerate(ori_actions):
                    if pbar is not None:
                        pbar.update()
                    obs, _, _, _, info = env.step(a)
                    #print(env.get_actors())
                    #print(obs["pointcloud"].keys())
                    #pointcloud = obs["pointcloud"]["xyzw"]
                    #segmentation = obs["pointcloud"]["Segmentation"]
                    #for col in range(segmentation.shape[1]):
                    #    print(f"Column {col} unique values: {np.unique(segmentation[:, col])}")
                    #from ManiSkill.data_collection.debug import plot_point_cloud
                    #from utils.filter_pointcloud import filter_pointcloud_by_segmentation
                    #obs = filter_pointcloud_by_segmentation(pointcloud, segmentation, [14])
                    #print(obs.shape)
                    #plot_point_cloud(obs)
                    #print(obs["pointcloud"]["xyzw"][-1])
                    #print(obs["pointcloud"]["Segmentation"][-1])
                    #quit()
                    if args.vis:
                        env.render_human()
                    if args.use_env_states:
                        env.set_state(ori_env_states[t])

            # From joint position to others
            elif ori_control_mode == "pd_joint_pos":
                info = from_pd_joint_pos(
                    target_control_mode,
                    ori_actions,
                    ori_env,
                    env,
                    render=args.vis,
                    pbar=pbar,
                    verbose=args.verbose,
                )

            # From joint delta position to others
            elif ori_control_mode == "pd_joint_delta_pos":
                info = from_pd_joint_delta_pos(
                    target_control_mode,
                    ori_actions,
                    ori_env,
                    env,
                    render=args.vis,
                    pbar=pbar,
                    verbose=args.verbose,
                )

            success = info.get("success", False)
            if args.discard_timeout:
                timeout = "TimeLimit.truncated" in info
                success = success and (not timeout)

            if success or args.allow_failure:
                env.flush_trajectory()
                env.flush_video()
                break
            else:
                # Rollback episode id for failed attempts
                env._episode_id -= 1
                if args.verbose:
                    print("info", info)
        else:
            tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")

    # Cleanup
    env.close()
    if ori_env is not None:
        ori_env.close()
    ori_h5_file.close()

    if pbar is not None:
        pbar.close()

    return output_h5_path


def main(args):
    if args.num_procs > 1:
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, args.num_procs) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        if args.save_traj:
            # A hack to find the path
            output_path = res[0][: -len("0.h5")] + "h5"
            merge_h5(output_path, res)
            for h5_path in res:
                tqdm.write(f"Remove {h5_path}")
                os.remove(h5_path)
                json_path = h5_path.replace(".h5", ".json")
                tqdm.write(f"Remove {json_path}")
                os.remove(json_path)
    else:
        _main(args)


if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main(parse_args())