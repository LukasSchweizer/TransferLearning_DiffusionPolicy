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
(*Starting with *TurnFaucet* and adding more later.*)

#### **Collect Demonstrations via Keyboard Teleop**
```bash
python ManiSkill/data_collection/teleop.py -e "TurnFaucet-v0"
```

#### **Download Demonstrations**
```bash
# Download all rigid-body demonstrations (TODO: Download only necessary demos)
gdown https://drive.google.com/drive/folders/1pd9Njg2sOR1VSSmp-c1mT7zCgJEnF8r7 --folder -O demos/
```
After downloading you have to unzip it into the demos folder

#### **Create .zarr from Demonstrations**
```bash
# Adapt file paths in train.py and run it
python train.py
```

#### **Collect Demonstrations via Prerecorded Trajectories**
```bash
# Save RGB
python -m mani_skill2.trajectory.replay_trajectory --traj-path "demos/TurnFaucet-v0/5000.h5" --vis --count 100 --save-traj -o "rgbd"

# Save PointCloud
python -m mani_skill2.trajectory.replay_trajectory --traj-path "demos/TurnFaucet-v0/5000.h5" --vis --count 100 --save-traj -o "pointcloud"
```

### 2. Train Diffusion Policy
```bash
# 2D Diffusion
python train.py

# 3D Diffusion
bash train_3dp.sh dp3 maniskill2_faucet 0322 0 0
```

### 3. Run "TurnFaucet-v0" as controlled the trained diffusion policy
```bash
# 2D Diffusion
python gym.py --object-id="5000"

# 3D Diffusion
bash eval_3dp.sh dp3 maniskill2_faucet 0322 0 0
```

### 4. Figure out how to transfer the policy to another action (5 more actions???)
```python
# TODO
```

## TODO:
1. Save videos/images from the teleop control
2. Record Demos (50-100)
3. Create trainer for Diffusion Policy
4. Incorporate Diffusion Policy
5. Tranfer learning to new tasks x5