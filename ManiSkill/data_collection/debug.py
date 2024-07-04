import matplotlib.pyplot as plt

# DO NOT CALL THESE FUNCTIONS INSIDE OF THE MAINLOOP OF MANISKILL, DUE TO CRASHES!!!

def plot_point_cloud(obs):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    tensor = obs["pointcloud"]["xyzw"][0]
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


def plot_rgb(obs, sensor_name: str = "base_camera"):
    image_tensor = obs["sensor_data"][sensor_name]["rgb"][0]
    fig = plt.imshow(image_tensor)
    plt.title(f'RGB Image from {sensor_name}')
    plt.show()
    plt.close()
