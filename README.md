# TransferLearning_DiffusionPolicy
Repository for the "Deep Learning Lab" offered by the University Freiburg. This project focusses on Transfer Learning with a learned Diffusion Policy.

*See docs/README.md for unabridged install and run instructions*

## Setup
With Miniconda installed on your machine, execute the setup script:
```bash
# Allow file execution
chmod +x conda_env_setup.sh

# Execute setup script
./conda_env_setup.sh

# Collect Faucet Assets
python -m mani_skill2.utils.download_asset partnet_mobility_faucet

# Install 3D Diffusion Policy (Point Cloud)
cd ..
git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git && cd 3D-Diffusion-Policy
cd 3D-Diffusion-Policy
pip install -e .
cd ../../TransferLearning_DiffusionPolicy

# Install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
You should now be in the activated conda environment, `dlproject`.

## Workflow: 
### 1. Collecting Demos

#### **Download Demonstrations**
```bash
# Download all rigid-body demonstrations (TODO: Download only necessary demos)
gdown https://drive.google.com/drive/folders/1pd9Njg2sOR1VSSmp-c1mT7zCgJEnF8r7 --folder -O demos/
```
After downloading you have to unzip it into the demos folder

#### **Generate Training Data via Prerecorded Trajectories**
- Place the downloaded trajectories (.h5 and .json) into the following file structure:
    - category_a will include any original training faucet models.
    - category_b will include any transfer faucet models (exclude if not needed)
- Feel free to change the experiment name, but do not change the names for the subdirectories (category_a and category_b).
```
demos
 ├── your_experiment          # Parent folder, specify as dataset path (change name, pass to train_3dp.py)
 |    ├── category_a          # First category of faucet, do not change this name
 |    |   └── *.h5, *.json    # Place any faucets from the first category in here (ex. 5000.h5, 5000.json)
 |    └── category_b          # First category of faucet, do not change this name
 |        └── *.h5, *.json    # Place any faucets from the second category in here
```

- Once you have added all desired models, run the following command at the root directory of the project:

```bash
# Generate demonstrations and save as .zarr
bash scripts/generate_data.sh "demos/your_experiment" 10 2
```

- Change the integer args to change number of trajectories to be collected for category_a and category_b respectively.
- This will save data into .zarr format. 
- Pass this zarr file location `demos/your_experiment/full_dataset.zarr` into the training script (or hydra file in maniskill2_faucet.yaml at dataset.zarr_path).

### 2. Train Diffusion Policy
```bash
# 3D Diffusion
bash scripts/train_3dp.sh dp3 maniskill2_faucet 0322 0 0 your_experiment
```

### 3. Run "TurnFaucet-v0" as controlled the trained diffusion policy
```bash
# 3D Diffusion
bash scripts/eval_3dp.sh dp3 maniskill2_faucet 0322 0 0
```


## Methodology:
- Single Faucet
    1. Single Faucet Original (without finetuning)
        ```
        demos
        ├── single_faucet_original          
        |    └── category_a          # 10 demos
        |        └── 5001  
        ```
    2. Single Faucet Transfer (without finetuning)
        ```
        demos
        ├── single_faucet_transfer         
        |    └── category_a             # 10 demos
        |        └── 5006   
        ```
    3. Single Faucet Finetuned
        ```
        demos
        ├── single_faucet_finetuned          
        |    ├── category_a             # 10 demos
        |    |   └── 5001    
        |    └── category_b             # 2 Demos
        |        └── 5006    
        ```
    - Results:
        |                  | single_faucet_original | single_faucet_transfer | single_faucet_finetuned |
        |------------------|------------------------|------------------------|-------------------------|
        |      success     |                        |                        |                         |
        | transfer success |                        |                        |                         |

- Single Category
    1. Single Category Original (without finetuning)
        ```
        demos
        ├── single_category_original         
        |    └── category_a                     # 10 demos
        |        └── 5001, 5037, 5064, 5025   
        ```
    2. Single Category Transfer (without finetuning)
        ```
        demos
        ├── single_category_transfer         
        |    ├── category_a                     # 10 demos
        |    |   └── 5001, 5037, 5064, 5025   
        |    └── category_b                     # 10 demos
        |        └── 5006    
        ```
    3. Single Category Finetuned
        ```
        demos
        ├── single_category_finetuned         
        |    ├── category_a                     # 10 demos
        |    |   └── 5001, 5037, 5064, 5025   
        |    └── category_b                     # 2 demos
        |        └── 5006    
        ```
    - Results:
        |                  | single_category_original | single_category_transfer | single_category_finetuned |
        |------------------|--------------------------|--------------------------|---------------------------|
        |      success     |             0.96         |            0.30          |            0.30           |
        | transfer success |             0.00         |            0.30          |        (0.30) 0.56        |
        |  average reward  |              -           |           0.3761         |           0.3762          |

- Multi-Category
    1. Multi-Category Original (without finetuning)
        ```
        demos
        ├── multi_category_original         
        |    └── category_a                         # 10 demos
        |        └── 5001, 5037, 5064, 5025, 5006    
        ```
    2. Multi-Category Transfer (without finetuning)
        ```
        demos
        ├── multi_category_transfer         
        |    ├── category_a                         # 5 demos
        |    |   └── 5001, 5037, 5064, 5025, 5006    
        |    └── category_b                         # 5 demos
        |        └── 5005, 5053, 5028, 5052    
        ```
    3. Multi-Category Finetuned
        ```
        demos
        ├── multi_category_transfer         
        |    ├── category_a                         # 5 demos
        |    |   └── 5001, 5037, 5064, 5025, 5006    
        |    └── category_b                         # 1 demos
        |        └── 5005, 5053, 5028, 5052    
        ```
    - Results
        |                  | multi_category_original | multi_category_transfer | multi_category_finetuned |
        |------------------|-------------------------|-------------------------|--------------------------|
        |      success     |                         |                         |                          |
        | transfer success |                         |                         |                          |

## Setup:
- Robot: Franka Robot
- Observation input: 
    - Robot State: joint position and tool control point (qpos+tcp)
    - Segmented pointcloud showing: robot, faucet
        - Point-cloud size: 1024
        - Point-cloud sampling method: Furthest-Point Sampling
- Epochs/Batchsize: 3000/128
- Control method: pd_joint_pos
- Segmented pointcloud: robot, faucet

