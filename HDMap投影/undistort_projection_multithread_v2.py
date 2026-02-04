#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDMap投影 V2 - 3D bbox → 2D bbox 投影（路侧标定版 lableRoadside）
按路侧标定文件中的camera ID进行坐标转换：
  世界坐标系(≈VirtualLidar) → virtualLidarToCam → 相机坐标系 → 图像坐标系
"""

import json
import yaml
import numpy as np
import cv2
from pathlib import Path
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import re

# 添加父目录到路径以导入 common_utils（使用绝对路径）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common_utils

warnings.filterwarnings('ignore', category=UserWarning)

# 颜色配置（20个类别 + unknown）
LABEL_COLORS = {
    "Car": [255, 0, 0],
    "Suv": [255, 69, 0],
    "Non_motor_rider": [255, 215, 0],
    "Bollards": [128, 128, 128],
    "Pedestrian": [0, 255, 0],
    "Crash_bucket": [192, 192, 192],
    "Tricycle": [135, 206, 250],
    "Truck": [255, 140, 0],
    "Motorcycle": [255, 0, 255],
    "Bus": [0, 0, 255],
    "Motor_rider": [255, 105, 180],
    "Pedestrian_else": [144, 238, 144],
    "Bicycle": [0, 255, 255],
    "Vehicle_else": [160, 82, 45],
    "Vehicle_door": [255, 192, 203],
    "Other_rider": [221, 160, 221],
    "Huge_vehicle": [139, 0, 0],
    "Unknown": [128, 128, 128],
    "Animal_small": [255, 228, 196],
    "Cone": [255, 255, 0],
    # 其他未知类别
    "unknown": [255, 0, 128]  # 紫红色，用于未定义的类别
}


def rodrigues_to_R(rvec3):
    """罗德里格斯向量转旋转矩阵"""
    r = np.asarray(rvec3, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(r)
    return R


def get_3d_bbox_corners(center, size, yaw):
    """计算3D bbox的8个角点（世界坐标系）"""
    x, y, z = center
    l, w, h = size

    # 局部坐标系的8个角点
    corners = np.array([
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
        [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2], [l/2, -w/2, h/2],
        [l/2, w/2, h/2], [-l/2, w/2, h/2]
    ])

    # 旋转矩阵（绕z轴）
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])

    # 旋转 + 平移
    corners_rotated = (R @ corners.T).T + center
    return corners_rotated


def find_roadside_gt_image(roadside_images_folder, cam_id, timestamp_ms, max_time_diff_ms=1000):
    """找到路侧相机GT图像

    根据cam_id找到对应的pinhole文件夹，然后查找最接近时间戳的图像

    Args:
        roadside_images_folder: 路侧图像根目录
        cam_id: 标定文件中的camera ID（如 "3", "6", "9", "0"）
        timestamp_ms: 时间戳（毫秒）
        max_time_diff_ms: 最大允许的时间差（毫秒）

    Returns:
        (image_path, time_diff) or (None, None)
    """
    roadside_images_folder = Path(roadside_images_folder)

    # cam_id到pinhole文件夹的映射
    cam_id_to_pinhole = {"3": "pinhole0", "6": "pinhole1", "9": "pinhole2", "0": "pinhole3"}
    pinhole_name = cam_id_to_pinhole.get(str(cam_id))

    if pinhole_name is None:
        # 尝试遍历所有pinhole文件夹找到匹配的cam_id
        for folder in roadside_images_folder.iterdir():
            if folder.is_dir() and folder.name.startswith("pinhole"):
                pattern = f"cam{cam_id}_*.png"
                files = list(folder.glob(pattern))
                if files:
                    pinhole_name = folder.name
                    break

    if pinhole_name is None:
        return None, None

    camera_folder = roadside_images_folder / pinhole_name
    if not camera_folder.exists():
        return None, None

    # 精确匹配
    expected_name = f"cam{cam_id}_{int(timestamp_ms)}.png"
    img_path = camera_folder / expected_name
    if img_path.exists():
        return img_path, 0

    # 模糊匹配：找最接近时间戳的图像
    pattern = f"cam{cam_id}_*.png"
    png_files = list(camera_folder.glob(pattern))

    if not png_files:
        return None, None

    closest_file = None
    min_diff = float('inf')

    for png_file in png_files:
        match = re.search(r'_(\d+)\.png$', png_file.name)
        if match:
            file_timestamp = int(match.group(1))
            diff = abs(file_timestamp - timestamp_ms)
            if diff < min_diff:
                min_diff = diff
                closest_file = png_file

    if closest_file and min_diff <= max_time_diff_ms:
        return closest_file, min_diff

    return None, min_diff if closest_file else None


class HDMapProjectorMultiThread:
    def __init__(self, roadside_calib_path, roadside_images_folder, camera_id, transforms=None):
        """
        初始化HDMap投影器（路侧标定版）

        Args:
            roadside_calib_path: 路侧标定文件路径 (calib.json)
            roadside_images_folder: 路侧图像文件夹路径
            camera_id: 标定文件中的camera ID（如 "3", "6", "9", "0"）
            transforms: world2lidar 变换矩阵列表（可选，路侧模式下不需要）
        """
        with open(roadside_calib_path, 'r') as f:
            self.roadside_calib = json.load(f)

        self.roadside_images_folder = Path(roadside_images_folder)
        self.camera_id = str(camera_id)
        self.transforms = transforms

        # 验证camera ID存在
        available_ids = list(self.roadside_calib.get("camera", {}).keys())
        if self.camera_id not in available_ids:
            raise ValueError(
                f"Camera ID '{self.camera_id}' 不存在于标定文件中。"
                f"可用的ID: {', '.join(available_ids)}"
            )

        # 加载指定camera的标定参数
        self.cam_params = self._load_camera_params(self.camera_id)

        # 设置OpenCV线程数
        cv2.setNumThreads(0)

    def _load_camera_params(self, cam_id):
        """加载路侧标定文件中指定camera ID的参数

        Args:
            cam_id: 标定文件中的camera ID

        Returns:
            包含K, D, R_V2C, t_V2C, is_fisheye, resolution的字典
        """
        cam_config = self.roadside_calib["camera"][cam_id]

        K = np.asarray(cam_config["intri"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(cam_config.get("distor", []), dtype=np.float64).reshape(-1) \
            if "distor" in cam_config else None
        is_fisheye = bool(cam_config.get("isFish", 0))

        R_V2C = rodrigues_to_R(cam_config["virtualLidarToCam"]["rotate"])
        t_V2C = np.asarray(cam_config["virtualLidarToCam"]["trans"],
                           dtype=np.float64).reshape(3, 1)

        if is_fisheye:
            resolution = tuple(self.roadside_calib["imgSize"]["fish"])
        else:
            resolution = tuple(self.roadside_calib["imgSize"]["notFish"])

        params = {
            'K': K,
            'D': D,
            'R_V2C': R_V2C,
            't_V2C': t_V2C,
            'is_fisheye': is_fisheye,
            'resolution': resolution,
            'cam_id': cam_id
        }

        print(f"✓ 加载路侧Camera {cam_id} 标定参数:")
        print(f"   分辨率: {resolution}")
        print(f"   鱼眼: {'是' if is_fisheye else '否'}")

        return params

    def project_bbox_to_roadside_camera(self, bbox_corners_world):
        """
        将3D bbox投影到路侧相机并计算2D bbox

        坐标变换：世界坐标(≈VirtualLidar) → virtualLidarToCam → 相机坐标 → 图像坐标

        Args:
            bbox_corners_world: 3D bbox的8个角点（世界坐标系）(8, 3)

        Returns:
            bbox_2d: [x1, y1, x2, y2] or None
            corners_2d: 投影后的角点坐标列表
        """
        K = self.cam_params['K']
        D = self.cam_params['D']
        R_V2C = self.cam_params['R_V2C']
        t_V2C = self.cam_params['t_V2C']
        is_fisheye = self.cam_params['is_fisheye']
        img_w, img_h = self.cam_params['resolution']

        # 步骤1: 世界坐标(≈VirtualLidar) → 相机坐标系
        points_cam = (R_V2C @ bbox_corners_world.T).T + t_V2C.T

        # 过滤相机前方的点
        valid_mask = points_cam[:, 2] > 0.1
        if not valid_mask.any():
            return None, None

        points_valid = points_cam[valid_mask]

        # 步骤2: 相机坐标系 → 图像坐标系
        if D is not None and len(D) > 0:
            rvec = np.zeros(3)
            tvec = np.zeros(3)

            if is_fisheye and len(D) >= 4:
                uv, _ = cv2.fisheye.projectPoints(
                    points_valid.reshape(-1, 1, 3),
                    rvec, tvec, K, D[:4]
                )
            else:
                uv, _ = cv2.projectPoints(
                    points_valid.reshape(-1, 1, 3),
                    rvec, tvec, K, D
                )
            uv = uv.reshape(-1, 2)
        else:
            # 无畸变，直接投影
            uv_homogeneous = (K @ points_valid.T).T
            uv = uv_homogeneous[:, :2] / uv_homogeneous[:, 2:3]

        # 计算2D bbox
        x1, y1 = uv.min(axis=0)
        x2, y2 = uv.max(axis=0)

        # 检查是否在图像内
        if x2 < 0 or y2 < 0 or x1 > img_w or y1 > img_h:
            return None, None

        # 裁剪到图像范围
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        bbox_2d = [float(x1), float(y1), float(x2), float(y2)]
        corners_2d = uv.tolist()

        return bbox_2d, corners_2d

    def process_single_frame(self, annotation_path, output_dir, timestamp_ms,
                           ego_vehicle_id=45, num_threads=1):
        """
        处理单帧数据（路侧标定版）

        Args:
            annotation_path: 标注文件路径
            output_dir: 输出目录
            timestamp_ms: 时间戳（毫秒）
            ego_vehicle_id: 自车ID（将被排除）
            num_threads: 线程数（路侧只有1个相机，此参数保留兼容）
        """
        output_dir = Path(output_dir)
        gt_dir = output_dir / "gt"
        overlay_dir = output_dir / "overlay"
        bbox_on_gt_dir = output_dir / "bbox_on_gt"

        gt_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        bbox_on_gt_dir.mkdir(parents=True, exist_ok=True)

        cam_id = self.camera_id
        img_w, img_h = self.cam_params['resolution']

        # 1. 加载标注
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # 2. 准备所有物体数据（排除自车）
        objects_data = []
        for obj in annotation.get('object', []):
            if obj['id'] == ego_vehicle_id:
                continue

            # 计算3D bbox的8个角点（世界坐标系）
            center = [obj['x'], obj['y'], obj['z']]
            size = [obj['length'], obj['width'], obj['height']]
            yaw = obj['yaw']
            bbox_corners = get_3d_bbox_corners(center, size, yaw)

            objects_data.append({
                'id': obj['id'],
                'label': obj['label'],
                'color': LABEL_COLORS.get(obj['label'], LABEL_COLORS['unknown']),
                'bbox_corners': bbox_corners,
                'bbox_3d': {
                    'x': obj['x'], 'y': obj['y'], 'z': obj['z'],
                    'length': obj['length'], 'width': obj['width'],
                    'height': obj['height'], 'yaw': obj['yaw']
                }
            })

        # 3. 查找路侧GT图像
        gt_img = None
        gt_image_path, time_diff = find_roadside_gt_image(
            self.roadside_images_folder, cam_id, timestamp_ms
        )
        if gt_image_path:
            gt_output = gt_dir / f"cam{cam_id}.jpg"
            img = cv2.imread(str(gt_image_path))
            if img is not None:
                cv2.imwrite(str(gt_output), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                gt_img = img
                if time_diff and time_diff > 0:
                    print(f"    GT图像时间差: {time_diff}ms")
        else:
            print(f"    ⚠️  Camera {cam_id}: 未找到时间戳 {timestamp_ms} 的GT图像")

        # 4. 投影所有物体的bbox到路侧相机
        bboxes = []
        for obj_data in objects_data:
            bbox_corners = obj_data['bbox_corners']
            bbox_2d, corners_2d = self.project_bbox_to_roadside_camera(bbox_corners)

            if bbox_2d is not None:
                bboxes.append({
                    'id': obj_data['id'],
                    'label': obj_data['label'],
                    'color': obj_data['color'],
                    'bbox_2d': bbox_2d,
                    'corners_2d': corners_2d,
                    'bbox_3d': obj_data['bbox_3d']
                })

        # 5. 生成纯bbox图（黑色背景 + 彩色bbox框）
        bbox_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        if bboxes:
            for bbox_info in bboxes:
                x1, y1, x2, y2 = [int(v) for v in bbox_info['bbox_2d']]
                color = tuple(int(c) for c in bbox_info['color'])
                cv2.rectangle(bbox_img, (x1, y1), (x2, y2), color, 2)

        overlay_output = overlay_dir / f"cam{cam_id}.jpg"
        cv2.imwrite(str(overlay_output), bbox_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # 6. 生成bbox投影在GT上的图
        if gt_img is not None:
            bbox_on_gt_img = gt_img.copy()

            if bboxes:
                for bbox_info in bboxes:
                    x1, y1, x2, y2 = [int(v) for v in bbox_info['bbox_2d']]
                    color = tuple(int(c) for c in bbox_info['color'])
                    cv2.rectangle(bbox_on_gt_img, (x1, y1), (x2, y2), color, 2)

            bbox_on_gt_output = bbox_on_gt_dir / f"cam{cam_id}.jpg"
            cv2.imwrite(str(bbox_on_gt_output), bbox_on_gt_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return True


def main():
    parser = argparse.ArgumentParser(description="HDMap投影 V2 - 3D bbox → 2D bbox（路侧标定版）")
    parser.add_argument("--roadside-calib", type=str, required=True,
                       help="路侧标定文件路径 (calib.json)")
    parser.add_argument("--roadside-images", type=str, required=True,
                       help="路侧图像文件夹路径")
    parser.add_argument("--camera-id", type=str, required=True,
                       help="标定文件中的camera ID (如 3, 6, 9, 0)")
    parser.add_argument("--annotation", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--timestamp", type=int, required=True)
    parser.add_argument("--ego-vehicle-id", type=int, default=45)

    args = parser.parse_args()

    projector = HDMapProjectorMultiThread(
        args.roadside_calib, args.roadside_images, args.camera_id
    )
    projector.process_single_frame(
        args.annotation, args.output_dir, args.timestamp,
        args.ego_vehicle_id
    )

if __name__ == "__main__":
    main()
