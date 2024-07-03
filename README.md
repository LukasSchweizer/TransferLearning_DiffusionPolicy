# TransferLearning_DiffusionPolicy
Repository for the "Deep Learning Lab" offered by the University Freiburg. This project focusses on Transfer Learning with a learned Diffusion Policy.

## Setup
With Miniconda installed on your machine, execute the setup script:
```bash
# Allow file execution
chmod +x dlproject_conda_env

# Execute setup script
./conda_env_setup.sh

# Collect Faucet Assets
python -m mani_skill.utils.download_asset partnet_mobility_faucet
```
You should now be in the activated conda environment, `dlproject`.

## Workflow: 
### 1. Collecting Demos
**Starting with *TurnFaucet* and adding more later.**

Collect Demonstrations via Keyboard Teleop
Navigate in the ```/Maniskill/data_collection``` folder and run the following command from your command line:
```bash
python teleop.py --env-id="AdaptedTurnFaucet-v1" --object-id="5001"
```

### 2. Train Diffusion Policy
```python
# TODO
```

### 3. Run "TurnFaucet-v1" as controlled the trained diffusion policy
```bash
# TODO: Add diffusion policy
cd ManiSkill
python gym.py "TurnFaucet-v1"
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