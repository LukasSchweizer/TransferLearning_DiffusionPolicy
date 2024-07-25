import argparse
import zarr
import os

from models.datasets.maniskill_dataset import ManiSkillTrajectoryDataset

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--replay-num", type=int, required=True)
    parser.add_argument(
        "--state-method",
        type=str,
        default="qpos_tcp",
        help="qpos, qpos_qpvel, qpos_tcp",
    )
    return parser.parse_args(args)

def main(args):
    # Save zarr to dataset folder
    zarr_path = os.path.join(args.directory, 'full_dataset.zarr')
    dataset = args.dataset_path[:-2] + 'pointcloud.pd_joint_pos.h5'
    print(dataset)

    trajectory_data = ManiSkillTrajectoryDataset(
        dataset_file=dataset, 
        load_count=args.replay_num, 
        success_only=True,
        device=None,
        zarr_path=zarr_path,
        state_method=args.state_method
    )

if __name__ == "__main__":
    main(parse_args())