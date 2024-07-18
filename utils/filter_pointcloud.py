import numpy as np


def filter_pointcloud_by_segmentation(pointcloud, segmentation, filter_classes, class_column=1):
    """
    Filters points in the point cloud based on the target classes in the segmentation data.

    :param pointcloud: numpy array of shape (N, 4), where N is the number of points.
                       Each point has 4 values (x, y, z, w).
    :param segmentation: numpy array of shape (N, 4), where N is the number of points.
                         Each point has 4 values corresponding to segmentation info.
    :param target_classes: list of integers representing the target classes to filter.
    :param class_column: integer representing the column index of the class information in segmentation.
                         Default is 0.
    :return: filtered_pointcloud: numpy array of shape (M, 4), where M is the number of points
                                  belonging to the target classes.
    """
    # Extract the class information from the specified column
    class_info = segmentation[:, class_column]

    # Create a mask for the points belonging to any of the target classes
    mask = np.isin(class_info, filter_classes)

    # Apply the mask to filter the point cloud
    filtered_pointcloud = pointcloud[~mask]

    return filtered_pointcloud


def downsample_point_clouds(point_clouds, num_points):
    """
    Downsample multiple point clouds to a fixed number of points using the same indices.

    Parameters:
    - point_clouds (list of np.array): List of point clouds to downsample. Each point cloud is a (N, D) array.
    - num_points (int): The number of points to downsample each point cloud to.

    Returns:
    - downsampled_point_clouds (list of np.array): List of downsampled point clouds.
    """
    # Ensure all point clouds have enough points
    min_points = min(pc.shape[0] for pc in point_clouds)
    if min_points < num_points:
        raise ValueError(f"One of the point clouds has fewer than {num_points} points.")

    # Select common indices for downsampling
    common_indices = np.random.choice(min_points, num_points, replace=False)

    # Downsample each point cloud using the common indices
    downsampled_point_clouds = [pc[common_indices, :] for pc in point_clouds]

    return downsampled_point_clouds
