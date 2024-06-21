import open3d as o3d
import numpy as np
import os
import numba


def load_terrain_map(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data['terrain']


@numba.njit
def downsample(points, voxel_size=0.10):
    sample_dict = dict()
    for i in range(points.shape[0]):
        point_coord = np.floor(points[i] / voxel_size)
        sample_dict[(int(point_coord[0]), int(point_coord[1]), int(point_coord[2]))] = True
    res = np.zeros((len(sample_dict), 3), dtype=np.float32)
    idx = 0
    for k, v in sample_dict.items():
        res[idx, 0] = k[0] * voxel_size + voxel_size / 2
        res[idx, 1] = k[1] * voxel_size + voxel_size / 2
        res[idx, 2] = k[2] * voxel_size + voxel_size / 2
        idx += 1
    return res


def visualize_point_cloud(points):
    # Create an Open3D point cloud from the numpy array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def main():
    terrain_map_folder = '/home/demo/Music/LiDAR_SOT/waymo_data/data/pc/terrain_pc'
    file_names = sorted(os.listdir(terrain_map_folder))

    # Set the interval for loading point clouds
    load_interval = 5  # Load every 5th point cloud
    voxel_size = 0.5  # Set the voxel size for downsampling

    for i, file_name in enumerate(file_names):
        # Only load and visualize every 'load_interval' point clouds
        if i % load_interval != 0:
            continue

        file_path = os.path.join(terrain_map_folder, file_name)
        points = load_terrain_map(file_path)

        # Downsample the point cloud
        downsampled_points = downsample(points, voxel_size)

        print(f"Visualizing {file_name} with {downsampled_points.shape[0]} points")
        visualize_point_cloud(downsampled_points)


if __name__ == '__main__':
    main()
