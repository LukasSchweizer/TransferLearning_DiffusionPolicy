# TransferLearning_DiffusionPolicy
Repository for the "Deep Learning Lab" offered by the University Freiburg. This project focusses on Transfer Learning with a learned Diffusion Policy.

## Setup
With Miniconda installed on your machine, execute the setup script:
```bash
# Allow file execution
chmod +x conda_env_setup.sh

# Execute setup script
./conda_env_setup.sh

# Collect Faucet Assets
python -m mani_skill2.utils.download_asset partnet_mobility_faucet

# Install Real-Standford Diffusion Policy (RGD/State)
cd ..
git clone https://github.com/real-stanford/diffusion_policy.git && cd diffusion_policy
pip install -e ../diffusion_policy

# Install 3D Diffusion Policy (Point Cloud)
cd ..
git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git && cd 3D-Diffusion-Policy
cd 3D-Diffusion-Policy
pip install -e .

#Install pointcloud visualizer adn pytorch3d
cd ../visualizer
pip install -e .
cd ../../TransferLearning_DiffusionPolicy

# Install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
You should now be in the activated conda environment, `dlproject`.

## Workflow: 
### 1. Collecting Demos

#### **Collect Demonstrations via Keyboard Teleop**
```bash
python ManiSkill/data_collection/teleop.py -e "TurnFaucet-v0"
```

#### **OR Download Demonstrations**
```bash
# Download all rigid-body demonstrations (TODO: Download only necessary demos)
gdown https://drive.google.com/drive/folders/1pd9Njg2sOR1VSSmp-c1mT7zCgJEnF8r7 --folder -O demos/
```
After downloading you have to unzip it into the demos folder

#### **Collect Demonstrations via Prerecorded Trajectories**
- In order to streamline the process of data preparation, place the downloaded trajectories (.h5 and .json) into the following file structure.
    - category_a will include any original training faucet models.
    - category_b will include any transfer faucet models (exclude if not needed)
- Feel free to change the experiment name, but do not change the names for the subdirectories (category_a and category_b).

```
demos
 ├── your_experiment          # Parent folder, specify as dataset path
 |    ├── category_a          # First category of faucet, do not change this name
 |    |   └── *.h5, *.json    # Place any faucets from the first category in here (ex. 5000.h5, 5000.json)
 |    └── category_b          # First category of faucet, do not change this name
 |        └── *.h5, *.json    # Place any faucets from the second category in here
```

- Once you have added all desired models, run the following command at the root directory of the project:

```bash
bash generate_data.sh "demos/your_experiment" 10 2
```

- Change the integer args to change number of trajectories to be collected for category_a and category_b respectively.
- This will save data into .zarr format. 
    - Pass this zarr file location `demos/your_experiment/full_dataset.zarr` into the training script (or hydra file in maniskill2_faucet.yaml at dataset.zarr_path).

- **Alternatively**, use the following commands to rerun trajectories for specific faucets:

```bash
# Save RGB
python -m mani_skill2.trajectory.replay_trajectory --traj-path "demos/TurnFaucet-v0/5000.h5" --vis --count 100 --save-traj -o "rgbd"
# Save PointCloud (only pointcloud)
python -m mani_skill2.trajectory.replay_trajectory --traj-path "demos/TurnFaucet-v0/5000.h5" --vis --count 100 --save-traj -o "pointcloud"
# with Segmentation
python -m ManiSkill.data_collection.replay_trajectory --traj-path "demos/TurnFaucet-v0/5000.h5" --vis --count 100 --save-traj -o "pointcloud"

# Generate Zarr File
python generate_data.py --dataset_path "demos/TurnFaucet-v0/5000.pointcloud.pd_joint_pos.h5" --directory "demos/TurnFaucet-v0" --replay_nums 100
```

### 2. Train Diffusion Policy
```bash
# 3D Diffusion
bash train_3dp.sh dp3 maniskill2_faucet 0322 0 0 your_experiment

# 2D Diffusion
python train.py
```

### 3. Run "TurnFaucet-v0" as controlled the trained diffusion policy
```bash
# 3D Diffusion
bash eval_3dp.sh dp3 maniskill2_faucet 0322 0 0

# 2D Diffusion
python gym.py --object-id="5000"
```


## Methodology:
1. ```DONE:``` Setup:
    | Proprioception | Point-cloud size | PC Sampling Method | Transfer-Success     | Same Faucet Success  | Epochs/Demos/Batch/Model  |
    |----------------|------------------|--------------------|----------------------|----------------------|---------------------------|
    | qpos + qvel    | 4096             | Uniform Sampling   | 0.0% (reward -)      | 15% (reward 0.1942)  | 200/100/16/5000           |
    | tcp            | 1024             | Furthest-Point     | 0.0% (reward 0.0570) | 5%  (reward 0.1074)  | 500/100/128/5000          |
    | qpos + qvel    | 1024             | Furthest-Point     | 0.0% (reward 0.0518) | 10% (reward 0.1537)  | 200/100/16/5000           |
    | qpos + qvel    | 1024             | Furthest-Point     | 0.0% (reward 0.0518) | 0% (reward 0.0575)   | 3000/10/128/5000          |
    | qpos + qvel    | 1024             | Furthest-Point     | 0.0% (reward 0.0518) | 5% (reward 0.0575)   | 200/40/128/5001           |
    | qpos + tcp     | 1024             | Furthest-Point     | -                    | 20% (reward 0.0575)  | 3000/10/128/5001          |
    | qpos + tcp     | 1024             | Furthest-Point     | -                    | 35% (reward 0.0575)  | 200/40/128/5001           |
    - Control method: pd_joint_pos
    - Segmented pointcloud: robot, faucet

2. ```DONE:``` Train 3DP on 1 Faucet, see if that is enough to transfer to one similarly categorized faucet 
    - Failed, overfits on the first faucet

    | Samples | Training-Faucets | Transfer-Faucet | Epochs | Transfer-Success |
    |---------|------------------|-----------------|--------|------------------|
    | 100     | 5000             | 5001            | 175    |                  |

3. ```TODO:``` Train 3DP on 1 Faucet, but use End Effector Position instead of pd_joint_pos, see if it transfers
4. ```TODO:``` Train 3DP on many examples of similar faucets, see if it can transfer to other faucets (similar category and other categories)
5. ```TODO:``` Train 3DP on many examples of multiplce categories of faucets, see if it can transfer to unseen categories
6. ```TODO:``` If previous successful, do step (4) on new tasks
    - Cabinet?
    - Peg Insert?
    - Object Pickup?