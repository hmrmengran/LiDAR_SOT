import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time


def load_and_inspect_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    print(f"Inspecting file: {file_path}")
    print(f"Keys in the .npz file: {data.files}")

    for key in data.files:
        value = data[key]
        print(f"Key: {key}")
        print(f"  Shape: {value.shape}")
        print(f"  Data Type: {value.dtype}")
        print(
            f"  Example Data: {value[:2] if value.size > 2 else value}")  # Print first 2 elements or all if less than 2

    return data


def visualize_data(data):
    # Visualize point cloud using Open3D
    if 'pc' in data:
        points = data['pc']
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([point_cloud])


def main():
    file_paths = ['/home/demo/Music/LiDAR_SOT/waymo_data/data/gt_shapes/-KpKb_7UDmsdQCf8FS3g9w.npz',
                  '/home/demo/Music/LiDAR_SOT/waymo_data/data/gt_shapes/XKFDjHhG0sIsQe0Jt89qJg.npz']

    for file_path in file_paths:
        data = load_and_inspect_npz(file_path)
        visualize_data(data)


if __name__ == '__main__':
    main()
