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
cd ../../TransferLearning_DiffusionPolicy
```
You should now be in the activated conda environment, `dlproject`.

## Workflow: 
### 1. Collecting Demos
**Starting with *TurnFaucet* and adding more later.**

Collect Demonstrations via Keyboard Teleop
Navigate in the ```/Maniskill/data_collection``` folder and run the following command from your command line:
```bash
python -m mani_skill2.trajectory.replay_trajectory --traj-path "demos/TurnFaucet-v0/5000.h5" --vis --count 100 --save-traj -o "rgbd"
```

### 2. Train Diffusion Policy
```bash
python train.py
```

### 3. Run "TurnFaucet-v0" as controlled the trained diffusion policy
```bash
python gym.py --object-id="5001"
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