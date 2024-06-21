import sys
import os
import json
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# Add the visualization directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'visualization'))

# Now you can import the visualizer3d module
from visualizer3d import VisualizerPangoV2
import pangolin
import sot_3d
from sot_3d.data_protos import BBox


def str_list_to_int(lst):
    result = []
    for t in lst:
        try:
            t = int(t)
            result.append(t)
        except:
            continue
    return result


def pc2world(ego_matrix, pcs):
    new_pcs = np.concatenate((pcs,
                              np.ones(pcs.shape[0])[:, np.newaxis]),
                             axis=1)
    new_pcs = ego_matrix @ new_pcs.T
    new_pcs = new_pcs.T[:, :3]
    return new_pcs


def create_shapes(pcs, bboxes, resolution=0.05):
    """ create the gt shape using input point clouds, bboxes, and ego transformations
    """
    assert len(pcs) == len(bboxes)
    shape_pc = np.zeros((0, 3))
    for i, (pc, bbox) in enumerate(zip(pcs, bboxes)):
        pc = sot_3d.utils.pc_in_box(bbox, pc, 1.0)
        bbox_state = BBox.bbox2array(bbox)[:4]
        pc -= bbox_state[:3]
        pc = sot_3d.utils.apply_motion_to_points(pc,
                                                 np.array([0, 0, 0, -bbox.o]))

        shape_pc = np.vstack((shape_pc, pc))
        # shape_pc = sot_3d.utils.downsample(shape_pc, voxel_size=resolution)
    return shape_pc


# Define data file paths
data_folder = '/home/demo/Music/LiDAR_SOT/waymo_data/data'
segment_name = 'segment-10203656353524179475_7625_000_7645_000_with_camera_labels'
tracking_result_file = os.path.join(data_folder, 'TrackingResults', 'debug', 'summary', '-KpKb_7UDmsdQCf8FS3g9w.json')
raw_pc_file = os.path.join(data_folder, 'pc', 'raw_pc', f'{segment_name}.npz')
clean_pc_file = os.path.join(data_folder, 'pc', 'clean_pc', f'{segment_name}.npz')
ego_info_file = os.path.join(data_folder, 'ego_info', f'{segment_name}.npz')

# Load data
bench_list_path = os.path.join(data_folder, 'bench_list.json')
with open(bench_list_path, 'r') as file:
    bench_list = json.load(file)
id = bench_list[0]['id']
segment_name = bench_list[0]['segment_name']
frame_range = bench_list[0]['frame_range']
print("id:", id, " segment_name:", segment_name, "frame_range: ", frame_range)

with open(tracking_result_file, 'r') as f:
    tracklet_results = json.load(f)

frame_keys = list(tracklet_results.keys())
print("frame_keys:", frame_keys)
frame_keys = sorted(str_list_to_int(frame_keys))
preds = list()
for key in frame_keys:
    bbox = tracklet_results[str(key)]['bbox1']
    bbox = BBox.array2bbox(bbox)
    preds.append(bbox)
preds.insert(0, BBox.array2bbox(tracklet_results[str(frame_keys[0])]['bbox0']))
print("preds:", preds)

scene_pcs = np.load(os.path.join(data_folder, 'pc', 'raw_pc', '{:}.npz'.format(segment_name)),
                    allow_pickle=True)
ego_infos = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)),
                    allow_pickle=True)
print("frame_range 0: ", frame_range[0], "frame_range 1: ", frame_range[1])
pcs = [pc2world(ego_infos[str(i)], scene_pcs[str(i)])
       for i in range(frame_range[0], frame_range[1] + 1)]
# shape = create_shapes(pcs, preds)


# Initialize visualizer tools
pc_visualizer = VisualizerPangoV2(name='Point Cloud Overlaid')


def visualize_pc_overlay(delay=0.1, initial_camera_params=None):
    try:
        assert len(pcs) == len(preds)
        accumulated_pc = np.empty((0, 3))
        for i, (pc, bbox) in enumerate(zip(pcs, preds)):
            pc = sot_3d.utils.pc_in_box(bbox, pc, 1.0)
            bbox_state = BBox.bbox2array(bbox)[:4]
            pc -= bbox_state[:3]
            pc = sot_3d.utils.apply_motion_to_points(pc, np.array([0, 0, 0, -bbox.o]))
            if i != 0:
                if np.linalg.norm(pc[0] - accumulated_pc[0]) < 0.3:
                    accumulated_pc = np.vstack((accumulated_pc, pc))
                    print(f"Frame {i}, last accumulated pc: {accumulated_pc[0]},  pc: {pc[0]}")
            else:
                accumulated_pc = np.vstack((accumulated_pc, pc))

            # Display the accumulated point cloud
            pc_visualizer.dq.put(
                ('reset', 'pts', 'all_pc', [accumulated_pc, {'Color': [0.0, 0.0, 1.0], 'PointSize': 1}]))
            # Set camera position only in the first frame
            if i == 0 and initial_camera_params:
                # Ensure look and up vectors are not parallel
                look_vector = initial_camera_params['look']
                up_vector = initial_camera_params['up']

                # Check if look and up vectors are parallel
                if np.dot(look_vector, up_vector) / (np.linalg.norm(look_vector) * np.linalg.norm(up_vector)) > 0.99:
                    up_vector = [0, 1, 0]  # Change up vector to ensure it's not parallel

                up_axis = pangolin.AxisDirection.AxisY if up_vector == [0, 1, 0] else pangolin.AxisDirection.AxisZ
                pc_visualizer.dq.put(('cam', 'cam', 'pos', {
                    'look': look_vector,
                    'at': initial_camera_params['at'],
                    'up': up_axis
                }))

            time.sleep(delay)
    except Exception as e:
        print(f"Error in visualize_pc_overlay: {e}")


# Start visualization
if __name__ == '__main__':
    initial_camera_params = {
        'look': [0, 0, 10],
        'at': [0, 0, 0],
        'up': [0, 0, 1]
    }
    visualize_pc_overlay(initial_camera_params=initial_camera_params)
    pc_visualizer.runner.join()
