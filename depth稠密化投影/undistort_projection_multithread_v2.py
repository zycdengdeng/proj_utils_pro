#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去畸变版投影：多线程CPU优化版 V2 - depth稠密化投影 (路侧标定版 lableRoadside)
新流程：世界坐标系 (≈VirtualLidar) → virtualLidarToCam (单个路侧相机) → 图像
输出深度图：近白远黑，无点区域纯黑色（.npy + .jpg）
深度图稠密化：形态学填充 → 引导滤波 → 最近邻插值 → 边缘保持平滑
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


def rodrigues_to_R(rvec3):
    """罗德里格斯向量转旋转矩阵"""
    r = np.asarray(rvec3, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(r)
    return R


def find_roadside_gt_image(roadside_images_folder, cam_id, timestamp_ms, max_time_diff_ms=1000):
    """找到路侧相机图像"""
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


class DepthDenseProjectorMultiThread:
    def __init__(self, roadside_calib_path, roadside_images_folder, camera_id):
        """
        初始化投影器

        Args:
            roadside_calib_path: 路侧标定文件路径
            roadside_images_folder: 路侧图像文件夹
            camera_id: 相机ID（如 "3", "6", "9", "0"）
        """
        with open(roadside_calib_path, 'r') as f:
            self.roadside_calib = json.load(f)

        self.roadside_images_folder = Path(roadside_images_folder)
        self.camera_id = str(camera_id)

        # 验证相机ID
        if self.camera_id not in self.roadside_calib["camera"]:
            available_ids = list(self.roadside_calib["camera"].keys())
            raise ValueError(f"相机ID '{self.camera_id}' 不存在于calib.json中。可用的相机ID: {available_ids}")

        # 加载相机参数
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

        # 设置OpenCV线程数
        cv2.setNumThreads(0)

    def densify_depth_image(self, depth_raw):
        """
        深度图稠密化处理（4级策略）- 向量化优化版本

        级别1: 形态学填充（小空洞 < 5x5）- 使用卷积计算邻域平均
        级别2: 邻域平均填充（中等空洞）- 使用卷积计算邻域平均
        级别3: 最近邻插值（大空洞）- 使用距离变换一次性完成
        级别4: 轻度平滑 - 使用双边滤波

        Args:
            depth_raw: 原始深度图 (float32)

        Returns:
            depth_dense: 稠密化后的深度图 (float32)
        """
        depth_dense = depth_raw.copy()
        valid_mask = depth_raw > 0

        if np.sum(valid_mask) == 0:
            return depth_dense

        h, w = depth_raw.shape

        # ====== 级别1: 形态学填充（小空洞）使用向量化 ======
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        depth_binary = (depth_dense > 0).astype(np.uint8) * 255
        depth_binary_closed = cv2.morphologyEx(depth_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 对新填充的区域，用3x3邻域平均填充（向量化版本）
        new_filled_mask = (depth_binary_closed > 0) & (depth_dense == 0)
        if np.any(new_filled_mask):
            # 创建一个权重矩阵，用于计算加权平均
            # 使用卷积计算邻域和与邻域计数
            kernel_3x3 = np.ones((3, 3), dtype=np.float32)

            # 计算邻域深度和
            depth_sum = cv2.filter2D(depth_dense, -1, kernel_3x3, borderType=cv2.BORDER_CONSTANT)

            # 计算邻域有效像素数（depth > 0 的像素）
            valid_binary = (depth_dense > 0).astype(np.float32)
            valid_count = cv2.filter2D(valid_binary, -1, kernel_3x3, borderType=cv2.BORDER_CONSTANT)

            # 计算平均值（避免除零）
            avg_depth = np.zeros_like(depth_dense)
            valid_avg_mask = valid_count > 0
            avg_depth[valid_avg_mask] = depth_sum[valid_avg_mask] / valid_count[valid_avg_mask]

            # 只填充需要填充的区域
            depth_dense[new_filled_mask] = avg_depth[new_filled_mask]

        # ====== 级别2: 5x5邻域平均填充（中等空洞）使用向量化 ======
        for iteration in range(2):
            depth_binary = (depth_dense > 0).astype(np.uint8) * 255
            depth_binary_dilated = cv2.dilate(depth_binary, np.ones((5, 5), np.uint8), iterations=1)
            new_filled_mask = (depth_binary_dilated > 0) & (depth_dense == 0)

            if np.any(new_filled_mask):
                # 使用5x5卷积核计算邻域平均（向量化版本）
                kernel_5x5 = np.ones((5, 5), dtype=np.float32)

                # 计算邻域深度和
                depth_sum = cv2.filter2D(depth_dense, -1, kernel_5x5, borderType=cv2.BORDER_CONSTANT)

                # 计算邻域有效像素数
                valid_binary = (depth_dense > 0).astype(np.float32)
                valid_count = cv2.filter2D(valid_binary, -1, kernel_5x5, borderType=cv2.BORDER_CONSTANT)

                # 计算平均值
                avg_depth = np.zeros_like(depth_dense)
                valid_avg_mask = valid_count > 0
                avg_depth[valid_avg_mask] = depth_sum[valid_avg_mask] / valid_count[valid_avg_mask]

                # 只填充需要填充的区域
                depth_dense[new_filled_mask] = avg_depth[new_filled_mask]

        # ====== 级别3: 最近邻插值（大空洞，距离<20像素）使用距离变换 ======
        invalid_mask = depth_dense == 0
        if np.any(invalid_mask):
            valid_mask_level3 = depth_dense > 0
            if np.any(valid_mask_level3):
                # 使用距离变换一次性找到所有无效像素的最近有效像素
                # distance_transform_edt 返回到最近有效像素的距离和索引
                distances, indices = ndimage.distance_transform_edt(
                    invalid_mask,
                    return_distances=True,
                    return_indices=True
                )

                # 只填充距离 < 20 像素的区域
                fill_mask = invalid_mask & (distances < 20)

                if np.any(fill_mask):
                    # indices[0] 是行索引，indices[1] 是列索引
                    nearest_y = indices[0][fill_mask]
                    nearest_x = indices[1][fill_mask]

                    # 从最近的有效像素复制深度值
                    depth_dense[fill_mask] = depth_dense[nearest_y, nearest_x]

        # ====== 级别4: 轻度双边滤波平滑 ======
        valid_mask_final = depth_dense > 0
        if np.sum(valid_mask_final) > 100:
            # 只对有效区域进行轻度平滑
            depth_temp = depth_dense.copy()
            depth_smoothed = cv2.bilateralFilter(
                depth_temp.astype(np.float32), d=5, sigmaColor=10, sigmaSpace=5
            )
            depth_dense[valid_mask_final] = depth_smoothed[valid_mask_final]

        # 保证无点区域仍为纯黑色
        depth_dense[depth_dense <= 0] = 0

        return depth_dense

    def project_to_camera_depth(self, points):
        """
        投影到相机平面并生成深度图

        深度图要求：近白远黑，无点区域纯黑色
        变换流程：世界坐标系 (≈VirtualLidar) → virtualLidarToCam → 图像坐标系
        """
        img_w, img_h = self.resolution

        # 变换：VirtualLidar → Camera
        points_cam = (self.R_V2C @ points.T).T + self.t_V2C.T

        # 过滤背后的点
        valid = points_cam[:, 2] > 0.1
        if not np.any(valid):
            depth_raw = np.zeros((img_h, img_w), dtype=np.float32)
            depth_vis = np.zeros((img_h, img_w), dtype=np.uint8)
            return depth_raw, depth_vis, 0

        points_valid = points_cam[valid]
        depths = points_valid[:, 2]  # z值即为深度

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

        # 过滤有效投影点
        valid_proj = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
                    (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
        uv_valid = uv[valid_proj].astype(int)
        depths_valid = depths[valid_proj]

        # 创建深度图（原始值）
        depth_raw = np.zeros((img_h, img_w), dtype=np.float32)

        if len(uv_valid) > 0:
            # 使用深度缓冲，保留最近的深度值
            for (u, v), depth in zip(uv_valid, depths_valid):
                if depth_raw[v, u] == 0 or depth < depth_raw[v, u]:
                    depth_raw[v, u] = depth

        # 进行稠密化处理
        depth_dense = self.densify_depth_image(depth_raw)

        # 创建可视化深度图：近白远黑
        depth_vis = np.zeros((img_h, img_w), dtype=np.uint8)
        valid_mask = depth_dense > 0

        if np.any(valid_mask):
            min_depth = np.min(depth_dense[valid_mask])
            max_depth = np.max(depth_dense[valid_mask])

            if max_depth > min_depth:
                # 近白远黑：depth越小，值越大（白色）
                depth_normalized = (max_depth - depth_dense[valid_mask]) / (max_depth - min_depth)
                depth_vis[valid_mask] = (depth_normalized * 255).astype(np.uint8)
            else:
                # 所有深度相同，设为中灰色
                depth_vis[valid_mask] = 128

        return depth_dense, depth_vis, len(uv_valid)

    def process_single_frame(self, pcd_path, output_dir, timestamp_ms):
        """
        处理单帧数据

        Args:
            pcd_path: PCD文件路径
            output_dir: 输出目录
            timestamp_ms: 时间戳（毫秒）
        """
        output_dir = Path(output_dir)
        depth_dir = output_dir / "depth"
        gt_dir = output_dir / "gt"
        compare_dir = output_dir / "compare"
        overlay_dir = output_dir / "overlay"

        depth_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        # 1. 加载点云
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)

        # 2. 投影生成深度图（带稠密化）
        depth_raw, depth_vis, count = self.project_to_camera_depth(points)

        # 3. 保存深度图（.npy原始值 + .jpg可视化）
        cam_name = f"cam{self.camera_id}"
        depth_npy_output = depth_dir / f"{cam_name}.npy"
        depth_jpg_output = depth_dir / f"{cam_name}.jpg"
        np.save(str(depth_npy_output), depth_raw)
        cv2.imwrite(str(depth_jpg_output), depth_vis, [cv2.IMWRITE_JPEG_QUALITY, 100])

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

        # 5. 生成compare图（GT和深度图左右对比）
        if gt_img is not None:
            depth_vis_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            compare_img = np.hstack([gt_img, depth_vis_color])
            compare_output = compare_dir / f"{cam_name}.jpg"
            cv2.imwrite(str(compare_output), compare_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # 6. 生成overlay图（深度图叠加到GT上，使用伪彩色）
        if gt_img is not None:
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            overlay_img = gt_img.copy()
            mask = depth_vis > 0
            overlay_img[mask] = depth_colored[mask]
            overlay_output = overlay_dir / f"{cam_name}.jpg"
            cv2.imwrite(str(overlay_output), overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return True


def main():
    parser = argparse.ArgumentParser(description="多线程优化去畸变版投影 V2 - Depth稠密化投影 (路侧标定版)")
    parser.add_argument("--roadside-calib", type=str, required=True)
    parser.add_argument("--roadside-images", type=str, required=True)
    parser.add_argument("--camera-id", type=str, required=True, help="路侧相机ID (如 3, 6, 9, 0)")
    parser.add_argument("--pcd", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--timestamp", type=int, required=True)

    args = parser.parse_args()

    projector = DepthDenseProjectorMultiThread(
        args.roadside_calib, args.roadside_images, args.camera_id
    )
    projector.process_single_frame(
        args.pcd, args.output_dir, args.timestamp
    )

if __name__ == "__main__":
    main()
