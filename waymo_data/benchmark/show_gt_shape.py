import numpy as np


def load_and_inspect_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    print(f"Inspecting file: {file_path}")
    print(f"Keys in the .npz file: {data.files}")

    for key in data.files:
        print(f"Key: {key}")
        value = data[key]
        print(f"  Shape: {value.shape}")
        print(f"  Data Type: {value.dtype}")
        print(
            f"  Example Data: {value[:5] if value.size > 5 else value}")  # Print first 5 elements or all if less than 5


def main():
    # Replace these with the actual paths to your .npz files
    file_paths = ['/home/demo/Music/LiDAR_SOT/waymo_data/data/gt_shapes/-KpKb_7UDmsdQCf8FS3g9w.npz',
                  '/home/demo/Music/LiDAR_SOT/waymo_data/data/gt_shapes/XKFDjHhG0sIsQe0Jt89qJg.npz']

    for file_path in file_paths:
        load_and_inspect_npz(file_path)


if __name__ == '__main__':
    main()
