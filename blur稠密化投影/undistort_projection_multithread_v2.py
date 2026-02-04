#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去畸变版投影：多线程CPU优化版 V2 - blur稠密化投影（路侧标定版 lableRoadside）
新流程：世界坐标系 (≈VirtualLidar) → virtualLidarToCam (单个路侧相机) → 图像
使用路侧相机给点云着色（所有4个路侧相机），然后投影到指定单个路侧相机，最后进行4级稠密化处理
"""

import json
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

warnings.filterwarnings('ignore', category=UserWarning)

# 路侧pinhole相机配置（用于着色）
ROADSIDE_CAMERAS = {
    0: {"name": "pinhole0", "cam_id": "3", "desc": "路侧相机3"},
    1: {"name": "pinhole1", "cam_id": "6", "desc": "路侧相机6"},
    2: {"name": "pinhole2", "cam_id": "9", "desc": "路侧相机9"},
    3: {"name": "pinhole3", "cam_id": "0", "desc": "路侧相机0"}
}


def rodrigues_to_R(rvec3):
    """罗德里格斯向量转旋转矩阵"""
    r = np.asarray(rvec3, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(r)
    return R


def find_roadside_image(roadside_images_folder, pinhole_name, cam_id, timestamp_ms, max_time_diff_ms=1000):
    """找到路侧相机图像（用于着色）"""
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


def find_roadside_gt_image(roadside_images_folder, cam_id, timestamp_ms, max_time_diff_ms=1000):
    """找到路侧相机图像（用于GT显示）"""
    roadside_images_folder = Path(roadside_images_folder)

    # 优先映射关系
    cam_id_to_pinhole = {"3": "pinhole0", "6": "pinhole1", "9": "pinhole2", "0": "pinhole3"}
    pinhole_name = cam_id_to_pinhole.get(str(cam_id))

    # 如果映射不存在，尝试搜索
    if pinhole_name is None:
        for folder in roadside_images_folder.iterdir():
            if folder.is_dir() and folder.name.startswith("pinhole"):
                files = list(folder.glob(f"cam{cam_id}_*.png"))
                if files:
                    pinhole_name = folder.name
                    break

    if pinhole_name is None:
        return None, None

    camera_folder = roadside_images_folder / pinhole_name
    if not camera_folder.exists():
        return None, None

    # 精确匹配
    img_path = camera_folder / f"cam{cam_id}_{int(timestamp_ms)}.png"
    if img_path.exists():
        return img_path, 0

    # 模糊匹配
    png_files = list(camera_folder.glob(f"cam{cam_id}_*.png"))
    if not png_files:
        return None, None

    closest_file, min_diff = None, float('inf')
    for f in png_files:
        m = re.search(r'_(\d+)\.png$', f.name)
        if m:
            diff = abs(int(m.group(1)) - timestamp_ms)
            if diff < min_diff:
                min_diff = diff
                closest_file = f

    if closest_file and min_diff <= max_time_diff_ms:
        return closest_file, min_diff

    return None, min_diff if closest_file else None


class BlurDenseProjectorMultiThread:
    def __init__(self, roadside_calib_path, roadside_images_folder, camera_id):
        """
        初始化投影器

        Args:
            roadside_calib_path: 路侧标定文件路径
            roadside_images_folder: 路侧图像文件夹路径
            camera_id: 投影目标相机ID（如 "3", "6", "9", "0"）
        """
        with open(roadside_calib_path, 'r') as f:
            self.roadside_calib = json.load(f)

        self.roadside_images_folder = Path(roadside_images_folder)
        self.camera_id = str(camera_id)

        # 验证相机ID
        if self.camera_id not in self.roadside_calib["camera"]:
            available_ids = list(self.roadside_calib["camera"].keys())
            raise ValueError(f"相机ID '{self.camera_id}' 不存在于calib.json中。可用的相机ID: {available_ids}")

        # 加载投影目标相机参数
        cam_config = self.roadside_calib["camera"][self.camera_id]

        self.K = np.asarray(cam_config["intri"], dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(cam_config.get("distor", []), dtype=np.float64).reshape(-1) if "distor" in cam_config else None
        self.is_fisheye = bool(cam_config.get("isFish", 0))

        self.R_V2C = rodrigues_to_R(cam_config["virtualLidarToCam"]["rotate"])
        self.t_V2C = np.asarray(cam_config["virtualLidarToCam"]["trans"], dtype=np.float64).reshape(3, 1)

        if self.is_fisheye:
            self.resolution = tuple(self.roadside_calib["imgSize"]["fish"])
        else:
            self.resolution = tuple(self.roadside_calib["imgSize"]["notFish"])

        # 加载所有4个路侧相机参数（用于着色）
        self.roadside_camera_params = {}
        for pinhole_id in range(4):
            self.load_roadside_camera_params(pinhole_id)

        # 设置OpenCV线程数
        cv2.setNumThreads(0)

    def load_roadside_camera_params(self, pinhole_id):
        """加载路侧相机参数（用于着色）"""
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

    def colorize_pointcloud_from_roadside(self, points, timestamp_ms):
        """使用路侧相机给点云着色（使用所有4个路侧相机）"""
        N = len(points)
        colors = np.zeros((N, 3), dtype=np.float32)
        color_counts = np.zeros(N, dtype=np.int32)

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

    def project_to_camera_with_densification(self, points, colors):
        """
        投影到相机平面并进行稠密化

        变换流程：世界坐标系 (≈VirtualLidar) → virtualLidarToCam → 图像坐标系
        """
        img_w, img_h = self.resolution

        # 变换：VirtualLidar → Camera
        points_cam = (self.R_V2C @ points.T).T + self.t_V2C.T

        # 过滤背后的点
        valid = points_cam[:, 2] > 0.1
        if not np.any(valid):
            return np.zeros((img_h, img_w, 3), dtype=np.uint8), 0

        points_valid = points_cam[valid]
        colors_valid = colors[valid] if colors is not None else None

        # 投影：Camera → Image
        if self.D is not None and len(self.D) > 0:
            rvec = np.zeros(3)
            tvec = np.zeros(3)

            if self.is_fisheye and len(self.D) >= 4:
                uv, _ = cv2.fisheye.projectPoints(
                    points_valid.reshape(-1, 1, 3),
                    rvec, tvec, self.K, self.D[:4]
                )
            else:
                uv, _ = cv2.projectPoints(
                    points_valid.reshape(-1, 1, 3),
                    rvec, tvec, self.K, self.D
                )
            uv = uv.reshape(-1, 2)
        else:
            uv_homogeneous = (self.K @ points_valid.T).T
            uv = uv_homogeneous[:, :2] / uv_homogeneous[:, 2:3]

        valid_proj = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
                    (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
        uv_valid = uv[valid_proj].astype(int)

        # 创建图像和深度缓冲
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        depth_buffer = np.full((img_h, img_w), np.inf, dtype=np.float32)

        if len(uv_valid) > 0:
            proj_colors = (colors_valid[valid_proj] * 255).astype(np.uint8)
            depths_valid = points_valid[valid_proj, 2]

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

    def process_single_frame(self, pcd_path, output_dir, timestamp_ms):
        """
        处理单帧数据

        Args:
            pcd_path: PCD文件路径
            output_dir: 输出目录
            timestamp_ms: 时间戳（毫秒）
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

        # 2. 使用路侧相机给点云着色（所有4个路侧相机）
        print(f"🎨 使用路侧相机着色点云...")
        colors = self.colorize_pointcloud_from_roadside(points, timestamp_ms)

        # 3. 投影点云到单个路侧相机（带稠密化）
        print(f"🔧 开始4级稠密化投影到cam{self.camera_id}...")
        proj_img, count = self.project_to_camera_with_densification(points, colors)

        cam_name = f"cam{self.camera_id}"
        proj_output = proj_dir / f"{cam_name}.jpg"
        cv2.imwrite(str(proj_output), proj_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # 4. 处理GT图像
        gt_image_path, time_diff = find_roadside_gt_image(
            self.roadside_images_folder, self.camera_id, timestamp_ms
        )

        gt_img = None
        if gt_image_path:
            gt_output = gt_dir / f"{cam_name}.jpg"
            img = cv2.imread(str(gt_image_path))
            if img is not None:
                cv2.imwrite(str(gt_output), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                gt_img = img
                if time_diff > 0:
                    print(f"  GT图像时间差: {time_diff}ms")

        # 5. 生成compare图（GT和PROJ左右对比）
        if gt_img is not None:
            compare_img = np.hstack([gt_img, proj_img])
            compare_output = compare_dir / f"{cam_name}.jpg"
            cv2.imwrite(str(compare_output), compare_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # 6. 生成overlay图（投影叠加到GT上）
        if gt_img is not None:
            overlay_img = gt_img.copy()
            mask = np.any(proj_img > 10, axis=2)
            overlay_img[mask] = proj_img[mask]
            overlay_output = overlay_dir / f"{cam_name}.jpg"
            cv2.imwrite(str(overlay_output), overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return True


def main():
    parser = argparse.ArgumentParser(description="多线程优化去畸变版投影 V2 - Blur稠密化投影 (路侧标定版)")
    parser.add_argument("--roadside-calib", type=str, required=True)
    parser.add_argument("--roadside-images", type=str, required=True)
    parser.add_argument("--camera-id", type=str, required=True, help="路侧相机ID (如 3, 6, 9, 0)")
    parser.add_argument("--pcd", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--timestamp", type=int, required=True)

    args = parser.parse_args()

    projector = BlurDenseProjectorMultiThread(
        args.roadside_calib, args.roadside_images, args.camera_id
    )
    projector.process_single_frame(
        args.pcd, args.output_dir, args.timestamp
    )

if __name__ == "__main__":
    main()
