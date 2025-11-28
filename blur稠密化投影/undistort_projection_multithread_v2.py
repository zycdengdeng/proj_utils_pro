#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去畸变版投影：多线程CPU优化版 V2 - blur稠密化投影（路侧着色+4级稠密化）
统一变换逻辑：世界坐标系 → LiDAR坐标系 → 相机坐标系
使用路侧相机给点云着色，然后投影到车端7个相机，最后进行4级稠密化处理
"""

import json
import yaml
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
import os
import re

# 添加父目录到路径以导入 common_utils（使用绝对路径）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common_utils

warnings.filterwarnings('ignore', category=UserWarning)

# 路侧pinhole相机配置
ROADSIDE_CAMERAS = {
    0: {"name": "pinhole0", "cam_id": "3", "desc": "路侧相机3"},
    1: {"name": "pinhole1", "cam_id": "6", "desc": "路侧相机6"},
    2: {"name": "pinhole2", "cam_id": "9", "desc": "路侧相机9"},
    3: {"name": "pinhole3", "cam_id": "0", "desc": "路侧相机0"}
}

# 车端相机配置
VEHICLE_CAMERAS = {
    1: {"name": "FN", "desc": "前视窄角30°", "resolution": (3840, 2160)},
    2: {"name": "FW", "desc": "前视广角120°", "resolution": (3840, 2160)},
    3: {"name": "FL", "desc": "左前视120°", "resolution": (3840, 2160)},
    4: {"name": "FR", "desc": "右前视120°", "resolution": (3840, 2160)},
    5: {"name": "RL", "desc": "左后视60°", "resolution": (1920, 1080)},
    6: {"name": "RR", "desc": "右后视60°", "resolution": (1920, 1080)},
    7: {"name": "RN", "desc": "后视60°", "resolution": (1920, 1080)}
}

def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵"""
    x, y, z, w = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])
    return R

def rodrigues_to_R(rvec3):
    """罗德里格斯向量转旋转矩阵"""
    r = np.asarray(rvec3, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(r)
    return R

def find_gt_image(gt_images_folder, camera_name, timestamp_ms):
    """找到最接近的真值图片"""
    camera_folder = Path(gt_images_folder) / camera_name
    if not camera_folder.exists():
        return None

    target_timestamp_us = timestamp_ms * 1000

    jpg_files = list(camera_folder.glob("*.jpg"))
    closest_file = None
    min_diff = float('inf')

    for jpg_file in jpg_files:
        match = re.search(r'_(\d+)\.(\d+)\.jpg$', jpg_file.name)
        if match:
            seconds = int(match.group(1))
            microseconds = int(match.group(2))
            timestamp_us = seconds * 1000000 + microseconds

            diff = abs(timestamp_us - target_timestamp_us)
            if diff < min_diff:
                min_diff = diff
                closest_file = jpg_file

    return closest_file

def find_roadside_image(roadside_images_folder, pinhole_name, cam_id, timestamp_ms, max_time_diff_ms=1000):
    """找到路侧相机图像"""
    camera_folder = Path(roadside_images_folder) / pinhole_name
    if not camera_folder.exists():
        return None, None

    expected_name = f"cam{cam_id}_{timestamp_ms}.png"
    img_path = camera_folder / expected_name

    if img_path.exists():
        return img_path, 0

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


class BlurDenseProjectorMultiThread:
    def __init__(self, roadside_calib_path, roadside_images_folder, vehicle_calib_folder,
                 gt_images_folder, transforms):
        """
        初始化投影器

        Args:
            roadside_calib_path: 路侧标定文件路径
            roadside_images_folder: 路侧图像文件夹路径
            vehicle_calib_folder: 车端标定文件夹路径
            gt_images_folder: 真值图像文件夹
            transforms: world2lidar 变换矩阵列表
        """
        with open(roadside_calib_path, 'r') as f:
            self.roadside_calib = json.load(f)
        self.roadside_images_folder = Path(roadside_images_folder)
        self.vehicle_calib_folder = Path(vehicle_calib_folder)
        self.gt_images_folder = Path(gt_images_folder)
        self.transforms = transforms
        self.roadside_camera_params = {}
        self.vehicle_camera_params = {}
        self.camera_poses = {}

        # 设置OpenCV线程数
        cv2.setNumThreads(0)

    def get_world2lidar_transform(self, timestamp_ms):
        """获取 world2lidar 变换矩阵"""
        transform = common_utils.find_closest_transform(timestamp_ms, self.transforms)

        if transform is None:
            raise ValueError(f"未找到时间戳 {timestamp_ms} 对应的变换矩阵")

        rotation = np.array(transform['world2lidar']['rotation']).reshape((3, 1))
        translation = np.array(transform['world2lidar']['translation']).reshape((3, 1))

        return rotation, translation

    def load_roadside_camera_params(self, pinhole_id):
        """加载路侧相机参数"""
        cam_id = ROADSIDE_CAMERAS[pinhole_id]["cam_id"]
        cam_config = self.roadside_calib["camera"][cam_id]

        K = np.asarray(cam_config["intri"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(cam_config.get("distor", []), dtype=np.float64).reshape(-1) if "distor" in cam_config else None
        is_fisheye = bool(cam_config.get("isFish", 0))

        R_V2C = rodrigues_to_R(cam_config["virtualLidarToCam"]["rotate"])
        t_V2C = np.asarray(cam_config["virtualLidarToCam"]["trans"], dtype=np.float64).reshape(3, 1)

        if is_fisheye:
            resolution = tuple(self.roadside_calib["imgSize"]["fish"])
        else:
            resolution = tuple(self.roadside_calib["imgSize"]["notFish"])

        self.roadside_camera_params[pinhole_id] = {
            'K': K,
            'D': D,
            'R_V2C': R_V2C,
            't_V2C': t_V2C,
            'is_fisheye': is_fisheye,
            'resolution': resolution,
            'cam_id': cam_id
        }

        return K, D, R_V2C, t_V2C, is_fisheye

    def load_vehicle_camera_params(self, cam_id):
        """加载车端相机参数"""
        with open(self.vehicle_calib_folder / f"camera_{cam_id:02d}_intrinsics.yaml", 'r') as f:
            intrinsics = yaml.safe_load(f)
        K = np.array(intrinsics['K']).reshape(3, 3)
        D = np.array(intrinsics['D'])

        with open(self.vehicle_calib_folder / f"camera_{cam_id:02d}_extrinsics.yaml", 'r') as f:
            extrinsics = yaml.safe_load(f)

        transform = extrinsics['transform']
        q = [transform['rotation']['x'], transform['rotation']['y'],
             transform['rotation']['z'], transform['rotation']['w']]
        t = np.array([transform['translation']['x'],
                     transform['translation']['y'],
                     transform['translation']['z']])
        R_cam = quaternion_to_rotation_matrix(q)

        self.camera_poses[cam_id] = {
            'R': R_cam,
            't': t,
            'label': extrinsics.get('label', ''),
            'name': VEHICLE_CAMERAS[cam_id]['name']
        }

        self.vehicle_camera_params[cam_id] = {
            'K': K,
            'D': D,
            'resolution': VEHICLE_CAMERAS[cam_id]["resolution"]
        }

        return K, D, R_cam, t

    def colorize_pointcloud_from_roadside(self, points, timestamp_ms):
        """使用路侧相机给点云着色"""
        N = len(points)
        colors = np.zeros((N, 3), dtype=np.float32)
        color_counts = np.zeros(N, dtype=np.int32)

        for pinhole_id in range(4):
            if pinhole_id not in self.roadside_camera_params:
                self.load_roadside_camera_params(pinhole_id)

        for pinhole_id in range(4):
            pinhole_name = ROADSIDE_CAMERAS[pinhole_id]['name']
            cam_id = ROADSIDE_CAMERAS[pinhole_id]['cam_id']

            img_path, time_diff = find_roadside_image(
                self.roadside_images_folder, pinhole_name, cam_id, timestamp_ms
            )
            if not img_path:
                print(f"  警告: 未找到{pinhole_name}的图像 (查找cam{cam_id}_{timestamp_ms}.png)")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  警告: 无法读取{img_path}")
                continue

            if time_diff > 0:
                print(f"  {pinhole_name}: 使用图像 {img_path.name} (时间差: {time_diff}ms)")

            params = self.roadside_camera_params[pinhole_id]
            K = params['K']
            D = params['D']
            R_V2C = params['R_V2C']
            t_V2C = params['t_V2C']
            is_fisheye = params['is_fisheye']
            img_w, img_h = params['resolution']

            points_vlidar = points
            points_cam = (R_V2C @ points_vlidar.T).T + t_V2C.T

            valid_mask = points_cam[:, 2] > 0.1
            if not np.any(valid_mask):
                continue

            valid_indices = np.where(valid_mask)[0]
            points_valid = points_cam[valid_mask]

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
                uv_homogeneous = (K @ points_valid.T).T
                uv = uv_homogeneous[:, :2] / uv_homogeneous[:, 2:3]

            valid_proj_mask = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
                             (uv[:, 1] >= 0) & (uv[:, 1] < img_h)

            if not np.any(valid_proj_mask):
                continue

            valid_proj_indices = valid_indices[valid_proj_mask]
            uv_valid = uv[valid_proj_mask].astype(int)

            for i, (u, v) in enumerate(uv_valid):
                point_idx = valid_proj_indices[i]
                bgr = img[v, u]
                rgb = bgr[::-1] / 255.0
                colors[point_idx] += rgb
                color_counts[point_idx] += 1

            print(f"  {pinhole_name}: 着色了 {len(uv_valid)} 个点")

        colored_mask = color_counts > 0
        colors[colored_mask] /= color_counts[colored_mask, np.newaxis]

        uncolored_count = np.sum(~colored_mask)
        if uncolored_count > 0:
            colors[~colored_mask] = 0.5
            print(f"  警告: {uncolored_count} 个点未被路侧相机着色（使用灰色）")

        colored_count = np.sum(colored_mask)
        print(f"  总计: {colored_count}/{N} 个点成功着色 ({colored_count/N*100:.1f}%)")

        return colors

    def undistort_gt_image(self, gt_image_path, cam_id, output_path):
        """对真值图像进行去畸变"""
        if not gt_image_path or not gt_image_path.exists():
            return False

        img = cv2.imread(str(gt_image_path))
        if img is None:
            return False

        K = self.vehicle_camera_params[cam_id]['K']
        D = self.vehicle_camera_params[cam_id]['D']
        w, h = self.vehicle_camera_params[cam_id]['resolution']

        if cam_id in [2, 3, 4] and np.max(np.abs(D)) > 1:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D[:4], (w, h), np.eye(3), balance=0.0
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D[:4], np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
            undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        else:
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
            undistorted = cv2.undistort(img, K, D, None, new_K)

        cv2.imwrite(str(output_path), undistorted, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return True

    def project_to_camera_with_densification(self, points, colors, rotate_world2lidar,
                                             trans_world2lidar, cam_id):
        """
        投影到去畸变的相机平面并进行稠密化

        变换流程：世界坐标系 → LiDAR坐标系 → 相机坐标系 → 图像坐标系
        """
        cam_info = VEHICLE_CAMERAS[cam_id]
        img_w, img_h = cam_info["resolution"]

        K = self.vehicle_camera_params[cam_id]['K']
        D = self.vehicle_camera_params[cam_id]['D']
        R_cam2lidar = self.camera_poses[cam_id]['R']
        t_cam2lidar = self.camera_poses[cam_id]['t']

        # 步骤1: 世界坐标系 → LiDAR坐标系
        points_lidar = common_utils.transform_points_to_lidar(
            points[:, :3],
            {'world2lidar': {
                'rotation': rotate_world2lidar.flatten().tolist(),
                'translation': trans_world2lidar.flatten().tolist()
            }}
        )

        # 步骤2: LiDAR坐标系 → 相机坐标系
        R_lidar2cam = R_cam2lidar.T
        t_lidar2cam = -R_cam2lidar.T @ t_cam2lidar

        points_cam = (R_lidar2cam @ points_lidar.T).T + t_lidar2cam

        # 过滤背后的点
        valid = points_cam[:, 2] > 0.1
        if not np.any(valid):
            return np.zeros((img_h, img_w, 3), dtype=np.uint8), 0

        points_valid = points_cam[valid]
        colors_valid = colors[valid] if colors is not None else None

        # 步骤3: 相机坐标系 → 图像坐标系（去畸变投影）
        if cam_id in [2, 3, 4] and np.max(np.abs(D)) > 1:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D[:4], (img_w, img_h), np.eye(3), balance=0.0
            )
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (img_w, img_h), 0, (img_w, img_h))

        uv_homogeneous = (new_K @ points_valid.T).T
        z_proj = uv_homogeneous[:, 2]
        uv = (uv_homogeneous[:, :2] / z_proj[:, np.newaxis]).astype(int)

        valid_proj = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
                    (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
        uv_valid = uv[valid_proj]

        # 创建图像和深度缓冲
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        depth_buffer = np.full((img_h, img_w), np.inf, dtype=np.float32)

        if len(uv_valid) > 0:
            proj_colors = (colors_valid[valid_proj] * 255).astype(np.uint8)
            depths_valid = z_proj[valid_proj]

            # 绘制点云并记录深度
            for (u, v), color, depth in zip(uv_valid, proj_colors, depths_valid):
                cv2.circle(img, (u, v), 2,
                         (int(color[2]), int(color[1]), int(color[0])), -1)
                if depth < depth_buffer[v, u]:
                    depth_buffer[v, u] = depth

        # 4级稠密化处理
        img_dense = self.densify_rgb_image(img, depth_buffer)

        return img_dense, len(uv_valid)

    def densify_rgb_image(self, rgb_image, depth_buffer, max_hole_size=100):
        """RGB图像4级稠密化和空洞填补"""

        # 创建有效像素掩码
        valid_mask = (rgb_image.sum(axis=2) > 0).astype(np.uint8)

        filled_rgb = rgb_image.copy()

        # 1. 小空洞：形态学填充
        filled_rgb = self._morphological_fill_rgb(filled_rgb, valid_mask, kernel_size=3)

        # 2. 中等空洞：引导滤波
        filled_rgb = self._guided_fill_rgb(filled_rgb, valid_mask, kernel_size=5)

        # 3. 大空洞：最近邻插值
        filled_rgb = self._nearest_neighbor_fill_rgb(filled_rgb, valid_mask, max_distance=10)

        # 4. 边缘保持平滑
        filled_rgb = self._edge_preserving_smooth_rgb(filled_rgb, valid_mask)

        return filled_rgb

    def _morphological_fill_rgb(self, rgb_image, valid_mask, kernel_size=3):
        """使用形态学操作填补RGB小空洞"""
        filled = rgb_image.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        for c in range(3):
            channel = filled[:, :, c]

            for _ in range(2):
                dilated = cv2.dilate(channel, kernel, iterations=1)
                mask = (valid_mask == 0) & (dilated > 0)
                channel[mask] = dilated[mask]

            filled[:, :, c] = channel

        return filled

    def _guided_fill_rgb(self, rgb_image, valid_mask, kernel_size=5):
        """使用引导滤波填补中等空洞"""
        filled = rgb_image.copy()

        gray = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY)

        for c in range(3):
            channel = filled[:, :, c].astype(np.float32)

            weight_mask = valid_mask.astype(np.float32)

            kernel = cv2.GaussianBlur(np.ones((kernel_size, kernel_size)),
                                         (kernel_size, kernel_size), 0)
            kernel = kernel / np.sum(kernel)

            channel_sum = cv2.filter2D(channel, -1, kernel)
            weight_sum = cv2.filter2D(weight_mask, -1, kernel)

            weight_sum[weight_sum == 0] = 1
            smooth_channel = channel_sum / weight_sum

            mask = (valid_mask == 0) & (smooth_channel > 0)
            channel[mask] = smooth_channel[mask]

            filled[:, :, c] = channel.astype(np.uint8)

        return filled

    def _nearest_neighbor_fill_rgb(self, rgb_image, valid_mask, max_distance=10):
        """使用最近邻填补大空洞"""
        h, w = valid_mask.shape
        filled = rgb_image.copy()

        valid_pixels = valid_mask > 0
        if not np.any(valid_pixels):
            return filled

        dist_transform, nearest_idx = ndimage.distance_transform_edt(
            ~valid_pixels, return_indices=True
        )

        fill_mask = (valid_mask == 0) & (dist_transform <= max_distance)

        for c in range(3):
            channel = filled[:, :, c]
            channel[fill_mask] = rgb_image[nearest_idx[0][fill_mask],
                                          nearest_idx[1][fill_mask], c]
            filled[:, :, c] = channel

        return filled

    def _edge_preserving_smooth_rgb(self, rgb_image, original_valid_mask):
        """边缘保持平滑"""
        smooth = cv2.bilateralFilter(
            rgb_image,
            d=5,
            sigmaColor=25,
            sigmaSpace=5
        )

        mask = original_valid_mask > 0
        blend_factor = 0.8

        for c in range(3):
            smooth[:, :, c][mask] = (
                blend_factor * rgb_image[:, :, c][mask] +
                (1 - blend_factor) * smooth[:, :, c][mask]
            )

        return smooth

    def process_single_camera(self, cam_id, points, colors, rotate_world2lidar,
                             trans_world2lidar, timestamp_ms, proj_dir, gt_dir,
                             compare_dir, overlay_dir):
        """处理单个相机（用于多线程）"""
        cam_name = VEHICLE_CAMERAS[cam_id]['name']

        results = {'cam_name': cam_name, 'proj_img': None, 'gt_img': None, 'count': 0}

        # 处理GT图像
        gt_image_path = find_gt_image(self.gt_images_folder, cam_name, timestamp_ms)
        if gt_image_path:
            gt_output = gt_dir / f"{cam_name}.jpg"
            if self.undistort_gt_image(gt_image_path, cam_id, gt_output):
                results['gt_img'] = cv2.imread(str(gt_output))

        # 投影点云（带稠密化）
        proj_img, count = self.project_to_camera_with_densification(
            points, colors, rotate_world2lidar, trans_world2lidar, cam_id
        )
        proj_output = proj_dir / f"{cam_name}.jpg"
        cv2.imwrite(str(proj_output), proj_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        results['proj_img'] = proj_img
        results['count'] = count

        # 生成compare图（GT和PROJ左右对比）
        if results['gt_img'] is not None and results['proj_img'] is not None:
            gt_img = results['gt_img']
            compare_img = np.hstack([gt_img, proj_img])
            compare_output = compare_dir / f"{cam_name}.jpg"
            cv2.imwrite(str(compare_output), compare_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # 生成overlay图（投影叠加到GT上）
        if results['gt_img'] is not None and results['proj_img'] is not None:
            gt_img = results['gt_img']
            overlay_img = gt_img.copy()
            mask = np.any(proj_img > 10, axis=2)
            overlay_img[mask] = proj_img[mask]
            overlay_output = overlay_dir / f"{cam_name}.jpg"
            cv2.imwrite(str(overlay_output), overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return results

    def process_single_frame(self, pcd_path, output_dir, timestamp_ms, num_threads=7):
        """
        处理单帧数据（多线程）

        Args:
            pcd_path: PCD文件路径
            output_dir: 输出目录
            timestamp_ms: 时间戳（毫秒）
            num_threads: 线程数
        """
        output_dir = Path(output_dir)
        proj_dir = output_dir / "proj"
        gt_dir = output_dir / "gt"
        compare_dir = output_dir / "compare"
        overlay_dir = output_dir / "overlay"

        proj_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        # 1. 加载点云
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)

        # 2. 获取 world2lidar 变换
        try:
            rotate_world2lidar, trans_world2lidar = self.get_world2lidar_transform(timestamp_ms)
        except ValueError as e:
            print(f"❌ {e}")
            return False

        # 3. 使用路侧相机给点云着色
        print(f"🎨 使用路侧相机着色点云...")
        colors = self.colorize_pointcloud_from_roadside(points, timestamp_ms)

        # 4. 加载车端相机参数
        for cam_id in range(1, 8):
            if cam_id not in self.vehicle_camera_params:
                self.load_vehicle_camera_params(cam_id)

        # 5. 多线程处理每个车端相机
        print(f"🔧 开始4级稠密化投影...")
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for cam_id in range(1, 8):
                future = executor.submit(
                    self.process_single_camera,
                    cam_id, points, colors, rotate_world2lidar,
                    trans_world2lidar, timestamp_ms, proj_dir, gt_dir,
                    compare_dir, overlay_dir
                )
                futures.append(future)

            for future in futures:
                result = future.result()

        return True


def main():
    parser = argparse.ArgumentParser(description="多线程优化去畸变版投影 V2 - Blur稠密化投影")
    parser.add_argument("--roadside-calib", type=str, required=True)
    parser.add_argument("--roadside-images", type=str, required=True)
    parser.add_argument("--vehicle-calib", type=str, required=True)
    parser.add_argument("--gt-images", type=str, required=True)
    parser.add_argument("--pcd", type=str, required=True)
    parser.add_argument("--transform-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--timestamp", type=int, required=True)
    parser.add_argument("--num-threads", type=int, default=7, help="每帧使用的线程数")

    args = parser.parse_args()

    transforms = common_utils.load_world2lidar_transforms(args.transform_json)

    projector = BlurDenseProjectorMultiThread(
        args.roadside_calib, args.roadside_images, args.vehicle_calib,
        args.gt_images, transforms
    )
    projector.process_single_frame(
        args.pcd, args.output_dir, args.timestamp, args.num_threads
    )

if __name__ == "__main__":
    main()
