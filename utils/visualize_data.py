from PIL import Image
import pickle
import gzip
import numpy as np
import zarr

def read_data(datasets_dir):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    #data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    data = zarr.load(datasets_dir)

    print(data)

    #X = np.array(data["agent"]["qpos"]).astype('float32')
    #y = np.array(data["actions"]).astype('float32')
    image = np.array(data[40]).astype('float32')

    # reshape images to be shape (720, 1280, 4)
    # wristimages = wristimages.reshape(-1, 480, 720, 4)
    # stationary = stationary.reshape(-1, 480, 720, 4)

    im1 = Image.fromarray(image.astype(np.uint8), mode='RGB')
    im1.show()

    # print(X)
    # print(X.shape)
    # print(y)
    # print(y.shape)
    print(image.shape)

if __name__ == "__main__":
    
    ##############################
    # Set the data location here
    data = 'demos/TurnFaucet-v0/5000.rgbd.pd_joint_pos_hoopla.zarr/data/img'
    ##############################

    read_data(data)