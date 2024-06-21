import sys
import os
import json
import numpy as np
import time

# 添加 visualization 目录到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'visualization'))

# 现在可以导入 visualizer3d 模块
from visualizer3d import VisualizerPangoV2
import pangolin
import OpenGL.GL as gl
from sot_3d.data_protos import BBox

# 定义数据文件路径
data_folder = '/home/demo/Music/LiDAR_SOT/waymo_data/data'
segment_name = 'segment-10203656353524179475_7625_000_7645_000_with_camera_labels'
tracking_result_file = os.path.join(data_folder, 'TrackingResults', 'debug', 'summary', '-KpKb_7UDmsdQCf8FS3g9w.json')
raw_pc_file = os.path.join(data_folder, 'pc', 'raw_pc', f'{segment_name}.npz')
clean_pc_file = os.path.join(data_folder, 'pc', 'clean_pc', f'{segment_name}.npz')
ego_info_file = os.path.join(data_folder, 'ego_info', f'{segment_name}.npz')


def get_ego_matrix(ego_info_entry):
    """
    将 ego_info 条目转换为 4x4 NumPy 数组
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
    # 对顶点按 z 坐标进行排序，将其分为底部和顶部两个部分
    sorted_indices = np.argsort(bbox_corners_local[:, 2])
    bbox_corners_local = bbox_corners_local[sorted_indices]

    # 将底部和顶部的点分开
    bottom_corners = bbox_corners_local[:4]
    top_corners = bbox_corners_local[4:]

    # 找到底部面的中心点
    bottom_center = np.mean(bottom_corners, axis=0)
    top_center = np.mean(top_corners, axis=0)

    def sort_corners(corners, center):
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        return corners[np.argsort(angles)]

    # 对底部和顶部的点按角度进行排序，以围成矩形
    bottom_corners = sort_corners(bottom_corners, bottom_center)
    top_corners = sort_corners(top_corners, top_center)

    # 重新排序顶点，以满足绘制 3D 边界框的要求
    reordered_corners = [
        bottom_corners[0], bottom_corners[1], bottom_corners[2], bottom_corners[3],  # 底部
        top_corners[0], top_corners[1], top_corners[2], top_corners[3]  # 顶部
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
    """transform box dict in waymo_open_format to array
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
    return pc - centroid


def get_current_camera_params(state):
    model_view_matrix = state.GetModelViewMatrix().m
    look = [model_view_matrix[0, 3], model_view_matrix[1, 3], model_view_matrix[2, 3]]
    at = [model_view_matrix[0, 2], model_view_matrix[1, 2], model_view_matrix[2, 2]]
    up = [model_view_matrix[0, 1], model_view_matrix[1, 1], model_view_matrix[2, 1]]
    return {'look': look, 'at': at, 'up': up}


# 加载数据
with open(tracking_result_file, 'r') as f:
    tracking_results = json.load(f)
raw_pc = np.load(raw_pc_file, allow_pickle=True)
clean_pc = np.load(clean_pc_file, allow_pickle=True)
ego_info = np.load(ego_info_file, allow_pickle=True)

# 初始化 OpenGlRenderState 和 Handler3D
state = pangolin.OpenGlRenderState(
    pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
    pangolin.ModelViewLookAt(0, -70, 70, 0, 0, 0, pangolin.AxisZ)
)
handler = pangolin.Handler3D(state)

# 初始化可视化工具
tracking_visualizer = VisualizerPangoV2(name='Tracking Result')
pc_visualizer = VisualizerPangoV2(name='Point Cloud Overlaid')

# 创建窗口并绑定Handler
pangolin.CreateWindowAndBind('Main', 640, 480)
pangolin.CreatePanel('ui').SetBounds(pangolin.Attach(0), pangolin.Attach(1), pangolin.Attach(0), pangolin.Attach(1))
display = pangolin.CreateDisplay().SetBounds(pangolin.Attach(0), pangolin.Attach(1), pangolin.Attach(0),
                                             pangolin.Attach(1), -640.0 / 480.0).SetHandler(handler)


def get_current_camera_params(state):
    model_view_matrix = state.GetModelViewMatrix().m
    look = [model_view_matrix[0, 3], model_view_matrix[1, 3], model_view_matrix[2, 3]]
    at = [model_view_matrix[0, 2], model_view_matrix[1, 2], model_view_matrix[2, 2]]
    up = [model_view_matrix[0, 1], model_view_matrix[1, 1], model_view_matrix[2, 1]]
    return {'look': look, 'at': at, 'up': up}


def visualize_pc_overlay(delay=0.1, initial_camera_params=None):
    try:
        previous_pc_world = np.empty((0, 3))  # 存储上一时刻的点云
        while not pangolin.ShouldQuit():
            for frame_idx, frame_data in tracking_results.items():
                ego_matrix = ego_info[str(frame_idx)]
                ego_matrix_inv = np.linalg.inv(ego_matrix)

                bbox = frame_data['bbox1']
                bbox_array = np.array(bbox)
                bbox_instance = BBox.array2bbox(bbox_array)
                bbox_local = transform_bbox_to_bbox_local(bbox_instance, ego_matrix_inv)
                pc = clean_pc[str(frame_idx)]

                box_array_local = BBox.bbox2array(bbox_local)
                pc_in_bbox = pc_in_box(box_array_local, pc, 1.0)
                pc_in_bbox = translate_point_cloud_to_origin(pc_in_bbox)

                # 累加点云
                accumulated_pc_world = np.vstack((previous_pc_world, pc_in_bbox))
                print(
                    f"Frame {frame_idx}, Accumulated PointCloud : {accumulated_pc_world.shape[0]}, Previous PointCloud : {previous_pc_world.shape[0]}, PointCloud : {pc_in_bbox.shape[0]}")

                # 显示累加的点云
                pc_visualizer.dq.put(
                    ('reset', 'pts', 'all_pc', [accumulated_pc_world, {'Color': [0.0, 0.0, 1.0], 'PointSize': 1}]))

                # 仅在第一帧设置相机位置
                if frame_idx == 0 and initial_camera_params:
                    pc_visualizer.dq.put(('cam', 'cam', 'pos', initial_camera_params))

                # 渲染
                display.Activate(state)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

                # 获取并打印当前相机参数
                current_camera_params = get_current_camera_params(state)
                print(f"Frame {frame_idx}, Current Camera Params: {current_camera_params}")

                time.sleep(delay)

                # 更新上一时刻的点云
                previous_pc_world = accumulated_pc_world

                pangolin.FinishFrame()
    except Exception as e:
        print(f"Error in visualize_pc_overlay: {e}")


if __name__ == '__main__':
    initial_camera_params = {
        'look': [0, 0, 10],  # 替换为你想要的look参数
        'at': [0, 0, 0],  # 替换为你想要的at参数
        'up': [0, 0, 1]  # 替换为你想要的up参数
    }

    # 使用新的初始相机参数
    visualize_pc_overlay(delay=1.0, initial_camera_params=initial_camera_params)
    tracking_visualizer.runner.join()
    pc_visualizer.runner.join()
