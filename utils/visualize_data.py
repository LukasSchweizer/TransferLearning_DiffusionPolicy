from PIL import Image
import pickle
import gzip
import numpy as np
import zarr
import visualizer
import matplotlib.pyplot as plt

def read_data(datasets_dir):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    #data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    data = zarr.load(datasets_dir)
    return data

def visualize_rgb(datasets_dir):
    data = read_data(datasets_dir)
    image = np.array(data[40]).astype('float32')
    print(image.shape)
    im1 = Image.fromarray(image.astype(np.uint8), mode='RGB')
    im1.show()

def plot_point_cloud(obs):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    tensor = obs[200]
    print(tensor.shape)
    x = tensor[:, 0]
    y = tensor[:, 1]
    z = tensor[:, 2]
    # Plot points
    img = ax.scatter(x, y, z)
    fig.colorbar(img)
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of XYZW Tensor')
    # Show plot
    plt.show()
    plt.close()

def visualize_pointcloud(datasets_dir):
    data = read_data(datasets_dir)
    # visualizer.visualize_pointcloud(data)
    plot_point_cloud(data)

if __name__ == "__main__":
    
    ##############################
    # Set the data location here
    data = 'demos/TurnFaucet-v0/5000.pointcloud.4096.pd_joint_pos.zarr/data/pointcloud'
    ##############################

    visualize_pointcloud(data)