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
from sot_3d.data_protos import BBox

# Define data file paths
data_folder = '/home/demo/Music/LiDAR_SOT/waymo_data/data'
segment_name = 'segment-10203656353524179475_7625_000_7645_000_with_camera_labels'
tracking_result_file = os.path.join(data_folder, 'TrackingResults', 'debug', 'summary', '-KpKb_7UDmsdQCf8FS3g9w.json')
raw_pc_file = os.path.join(data_folder, 'pc', 'raw_pc', f'{segment_name}.npz')
clean_pc_file = os.path.join(data_folder, 'pc', 'clean_pc', f'{segment_name}.npz')
ego_info_file = os.path.join(data_folder, 'ego_info', f'{segment_name}.npz')

# Load data
with open(tracking_result_file, 'r') as f:
    tracking_results = json.load(f)
raw_pc = np.load(raw_pc_file, allow_pickle=True)
clean_pc = np.load(clean_pc_file, allow_pickle=True)
ego_info = np.load(ego_info_file, allow_pickle=True)

# Initialize visualizer tools
tracking_visualizer = VisualizerPangoV2(name='Tracking Result')
pc_visualizer = VisualizerPangoV2(name='Point Cloud Overlaid')


def get_ego_matrix(ego_info_entry):
    """
    Convert an ego_info entry to a 4x4 NumPy array
    """
    return np.array(ego_info_entry).reshape(4, 4)


def transform_bbox_to_bbox_local(bbox, ego_matrix_inv):
    bbox_local = BBox.bbox2world(ego_matrix_inv, bbox)
    return bbox_local


def transform_bbox_to_corners_local(bbox, ego_matrix_inv):
    bbox_local = BBox.bbox2world(ego_matrix_inv, bbox)
    bbox_corners_local = np.array(BBox.box2corners3d(bbox_local))
    return bbox_corners_local


def reorder_corners(bbox_corners_local):
    # Sort vertices by z-coordinate and divide them into bottom and top parts
    sorted_indices = np.argsort(bbox_corners_local[:, 2])
    bbox_corners_local = bbox_corners_local[sorted_indices]

    # Separate bottom and top points
    bottom_corners = bbox_corners_local[:4]
    top_corners = bbox_corners_local[4:]

    # Find the center point of the bottom face
    bottom_center = np.mean(bottom_corners, axis=0)
    top_center = np.mean(top_corners, axis=0)

    def sort_corners(corners, center):
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        return corners[np.argsort(angles)]

    # Sort bottom and top points by angle to form a rectangle
    bottom_corners = sort_corners(bottom_corners, bottom_center)
    top_corners = sort_corners(top_corners, top_center)

    # Reorder vertices to meet the requirements of drawing a 3D bounding box
    reordered_corners = [
        bottom_corners[0], bottom_corners[1], bottom_corners[2], bottom_corners[3],  # Bottom
        top_corners[0], top_corners[1], top_corners[2], top_corners[3]  # Top
    ]

    return reordered_corners


def pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling=1.0):
    mask = np.zeros(pc.shape[0], dtype=np.int32)
    yaw_cos, yaw_sin = np.cos(yaw), np.sin(yaw)
    for i in range(pc.shape[0]):
        rx = np.abs((pc[i, 0] - center_x) * yaw_cos + (pc[i, 1] - center_y) * yaw_sin)
        ry = np.abs((pc[i, 0] - center_x) * -yaw_sin + (pc[i, 1] - center_y) * yaw_cos)
        rz = np.abs(pc[i, 2] - center_z)

        if rx < (length * box_scaling / 2) and ry < (width * box_scaling / 2) and rz < (height * box_scaling / 2):
            mask[i] = 1
    indices = np.argwhere(mask == 1)
    result = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for i in range(indices.shape[0]):
        result[i, :] = pc[indices[i], :]
    return result


def pc_in_box(box, pc, box_scaling=1.0):
    center_x, center_y, length, width = \
        box[0], box[1], box[4], box[5]
    center_z, height = box[2], box[6]
    yaw = box[3]
    return pc_in_box_inner(center_x, center_y, center_z, length, width, height, yaw, pc, box_scaling)


def pb2dict(obj):
    """
    Takes a ProtoBuf Message obj and converts it to a dict.
    """
    adict = {}

    # Iterate through all fields in the Protobuf message descriptor
    for field in obj.DESCRIPTOR.fields:
        # Skip fields that are not set
        if not getattr(obj, field.name):
            continue

        # Handle non-repeated fields
        if not field.label == FD.LABEL_REPEATED:
            # If the field is not a message type, add it directly to the dictionary
            if not field.type == FD.TYPE_MESSAGE:
                adict[field.name] = getattr(obj, field.name)
            else:
                # If the field is a message type, recursively convert it to a dictionary
                value = pb2dict(getattr(obj, field.name))
                if value:
                    adict[field.name] = value
        else:
            # Handle repeated fields (lists)
            if field.type == FD.TYPE_MESSAGE:
                # If the field is a repeated message type, recursively convert each item to a dictionary
                adict[field.name] = [pb2dict(v) for v in getattr(obj, field.name)]
            else:
                # If the field is not a message type, add the list of values directly to the dictionary
                adict[field.name] = [v for v in getattr(obj, field.name)]

    return adict


def bbox_dict2array(box_dict):
    """
    Transform box dict in waymo_open_format to array
    Args:
        box_dict ([dict]): waymo_open_dataset formatted bbox
    """
    result = np.array([
        box_dict['center_x'],
        box_dict['center_y'],
        box_dict['center_z'],
        box_dict['heading'],
        box_dict['length'],
        box_dict['width'],
        box_dict['height']
    ])
    return result


def translate_point_cloud_to_origin(pc):
    centroid = np.mean(pc, axis=0)
    # centroid = np.array([7218.0, -1591.0, 213.0])
    # print("centroid:", centroid)
    return pc - centroid


def transform_point_cloud_to_world(pc, ego_matrix):
    """
    Transform the point cloud from the local coordinate system to the world coordinate system
    """
    pc_homogeneous = np.hstack((pc, np.ones((pc.shape[0], 1))))
    pc_world = np.dot(pc_homogeneous, ego_matrix.T)[:, :3]
    return pc_world


def transform_point_cloud(pc, motion):
    """
    Transform the point cloud based on the motion information
    """
    delta_x, delta_y, delta_z, delta_heading = motion
    # Generate rotation matrix around the Z axis
    rotation = R.from_euler('z', delta_heading).as_matrix()
    # Apply rotation and translation to the point cloud
    transformed_pc = np.dot(pc, rotation.T) + np.array([delta_x, delta_y, delta_z])
    return transformed_pc


def visualize_tracking(delay=0.1):
    try:
        for frame_idx, frame_data in tracking_results.items():
            bbox = frame_data['bbox1']
            pc = raw_pc[str(frame_idx)]
            ego_matrix_inv = np.linalg.inv(get_ego_matrix(ego_info[str(frame_idx)]))

            # print(f"Frame {frame_idx}, bbox type: {type(bbox)}, bbox: {bbox}")
            bbox_instance = BBox.array2bbox(np.array(bbox))
            # bbox_local = transform_bbox_to_bbox_local(bbox_instance, ego_matrix_inv)
            # bbox_corners_local_ordered = reorder_corners(bbox_corners_local)
            # print(f"Frame {frame_idx}, bbox_local: {bbox_local}")

            bbox_corners_local = transform_bbox_to_corners_local(bbox_instance, ego_matrix_inv)
            bbox_corners_local_ordered = reorder_corners(bbox_corners_local)
            # print(f"Frame {frame_idx}, bbox_corners_local: {bbox_corners_local}")
            # print(f"Frame {frame_idx}, bbox_corners_local_ordered: {bbox_corners_local_ordered}")
            # print('\n')

            # if pc.shape[1] > 3:  # Modify: Assume point cloud data contains intensity information
            #     intensity = pc[:, 3]
            #     colors = np.vstack([intensity, intensity, intensity]).T
            #     colors = colors / colors.max()  # Modify: Normalize to 0-1 range
            # else:
            #     colors = np.ones_like(pc[:, :3]) * [0.0, 1.0, 0.0]  # Modify: Set to green if no intensity information
            # points_with_colors = np.hstack((pc[:, :3], colors))
            tracking_visualizer.dq.put(('reset', 'pts', 'pc', [pc, {'Color': [0.0, 1.0, 0.0], 'PointSize': 2}]))

            tracking_visualizer.dq.put(
                ('reset', 'boxes_3d', 'bbox', [[bbox_corners_local_ordered], {'Color': [1.0, 0.0, 0.0]}]))
            # tracking_visualizer.dq.put(('reset', 'boxes_3d_center', 'bbox', [
            #     [[bbox_local.x, bbox_local.y, bbox_local.z, bbox_local.w, bbox_local.l, bbox_local.h, bbox_local.o]],
            #     {'Color': [1.0, 0.0, 0.0]}]))
            tracking_visualizer.dq.put(
                ('cam', 'cam', 'pos', {'look': [0, -70, 70], 'at': [0, 0, 0], 'up': pangolin.AxisZ}))
            time.sleep(delay)
    except Exception as e:
        print(f"Error in visualize_tracking: {e}")


def visualize_pc_overlay(delay=0.1, initial_camera_params=None):
    try:
        # Store the point cloud from the previous moment
        # previous_pc_world = np.empty((0, 3))
        accumulated_pc_world = np.empty((0, 3))
        for frame_idx, frame_data in tracking_results.items():
            ego_matrix = ego_info[str(frame_idx)]
            ego_matrix_inv = np.linalg.inv(ego_matrix)

            bbox = frame_data['bbox1']
            motion = frame_data['motion']

            # print(f"Frame {frame_idx}, bbox type: {type(bbox)}, bbox: {bbox}")
            bbox_array = np.array(bbox)
            bbox_instance = BBox.array2bbox(bbox_array)
            bbox_local = transform_bbox_to_bbox_local(bbox_instance, ego_matrix_inv)
            # print(f"Frame {frame_idx}, bbox type: {type(bbox_local)}, bbox_local: {bbox_local}")
            pc = clean_pc[str(frame_idx)]

            box_array_local = BBox.bbox2array(bbox_local)
            # print(f"Frame {frame_idx}, bbox type: {type(box_array_local)}, box_array_local: {box_array_local}")
            pc_in_bbox = pc_in_box(box_array_local, pc, 1.0)

            pc_in_bbox_world = transform_point_cloud_to_world(pc_in_bbox, ego_matrix)
            # print('ego_matrix', ego_matrix)
            print(f"Frame {frame_idx}, pc_in_bbox_world", pc_in_bbox_world[0])

            # Accumulate point cloud
            if int(frame_idx) > 1:
                # Transform the previous point cloud to the current coordinate system
                accumulated_pc_world_motion = transform_point_cloud(accumulated_pc_world, motion)
                print(f"Frame {frame_idx}, motion", motion)
                print(f"Frame {frame_idx}, accumulated_pc_world motion", accumulated_pc_world_motion[0])
                accumulated_pc_world = np.vstack((accumulated_pc_world_motion, pc_in_bbox_world))
            else:
                accumulated_pc_world = pc_in_bbox_world

            accumulated_pc_world_render = translate_point_cloud_to_origin(accumulated_pc_world)
            # Display the accumulated point cloud
            pc_visualizer.dq.put(
                ('reset', 'pts', 'all_pc', [accumulated_pc_world_render, {'Color': [0.0, 0.0, 1.0], 'PointSize': 1}]))
            # Set camera position only in the first frame
            if frame_idx == 0 and initial_camera_params:
                pc_visualizer.dq.put(('cam', 'cam', 'pos', initial_camera_params))

            time.sleep(delay)

            # # Update the point cloud from the previous moment
            # accumulated_pc_world = accumulated_pc_world
            if int(frame_idx) > 2:
                continue
    except Exception as e:
        print(f"Error in visualize_pc_overlay: {e}")


# Start visualization
if __name__ == '__main__':
    initial_camera_params = {
        'look': [0, 0, 10],
        'at': [0, 0, 0],
        'up': [0, 0, 1]
    }
    visualize_tracking(delay=0.5)
    visualize_pc_overlay(delay=1.0, initial_camera_params=initial_camera_params)
    tracking_visualizer.runner.join()
    pc_visualizer.runner.join()
