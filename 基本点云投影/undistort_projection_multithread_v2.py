#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本点云投影 V2 - 点云投影到路侧相机（路侧标定版 lableRoadside）
按路侧标定文件中的camera ID进行坐标转换：
  世界坐标系(≈VirtualLidar) → virtualLidarToCam → 相机坐标系 → 图像坐标系
"""

import json
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import argparse
import warnings
import sys
import os
import re

# 添加父目录到路径以导入 common_utils（使用绝对路径）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common_utils

warnings.filterwarnings('ignore', category=UserWarning)


def rodrigues_to_R(rvec3):
    """罗德里格斯向量转旋转矩阵"""
    r = np.asarray(rvec3, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(r)
    return R


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


class UndistortProjectorMultiThread:
    def __init__(self, roadside_calib_path, roadside_images_folder, camera_id):
        """
        初始化投影器（路侧标定版）

        Args:
            roadside_calib_path: 路侧标定文件路径 (calib.json)
            roadside_images_folder: 路侧图像文件夹路径
            camera_id: 标定文件中的camera ID（如 "3", "6", "9", "0"）
        """
        with open(roadside_calib_path, 'r') as f:
            self.roadside_calib = json.load(f)

        self.roadside_images_folder = Path(roadside_images_folder)
        self.camera_id = str(camera_id)

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

    def project_to_camera_undistorted(self, points, colors):
        """
        投影到路侧相机平面

        坐标变换：世界坐标系(≈VirtualLidar) → virtualLidarToCam → 相机坐标系 → 图像坐标系

        Args:
            points: 点云坐标 (N, 3)
            colors: 点云颜色 (N, 3)

        Returns:
            img: 投影图像
            count: 投影点数量
        """
        K = self.cam_params['K']
        D = self.cam_params['D']
        R_V2C = self.cam_params['R_V2C']
        t_V2C = self.cam_params['t_V2C']
        is_fisheye = self.cam_params['is_fisheye']
        img_w, img_h = self.cam_params['resolution']

        # 步骤1: 世界坐标(≈VirtualLidar) → 相机坐标系
        points_cam = (R_V2C @ points[:, :3].T).T + t_V2C.T

        # 过滤背后的点
        valid = points_cam[:, 2] > 0.1
        if not np.any(valid):
            return np.zeros((img_h, img_w, 3), dtype=np.uint8), 0

        points_valid = points_cam[valid]
        colors_valid = colors[valid] if colors is not None else None

        # 步骤2: 相机坐标系 → 图像坐标系（投影）
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

        # 转换为整数像素坐标
        uv = uv.astype(int)

        # 过滤有效投影点
        valid_proj = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
                    (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
        uv_valid = uv[valid_proj]

        # 创建图像
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        if len(uv_valid) > 0 and colors_valid is not None:
            proj_colors = (colors_valid[valid_proj] * 255).astype(np.uint8)
            # 使用cv2.circle绘制半径为2的圆
            for (u, v), color in zip(uv_valid, proj_colors):
                cv2.circle(img, (u, v), 2,
                         (int(color[2]), int(color[1]), int(color[0])), -1)

        return img, len(uv_valid)

    def process_single_frame(self, pcd_path, output_dir, timestamp_ms):
        """
        处理单帧数据（路侧标定版）

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

        cam_id = self.camera_id
        img_w, img_h = self.cam_params['resolution']

        # 1. 加载点云
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(points), 3)) * 0.5

        # 2. 投影点云到路侧相机
        proj_img, count = self.project_to_camera_undistorted(points, colors)
        proj_output = proj_dir / f"cam{cam_id}.jpg"
        cv2.imwrite(str(proj_output), proj_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

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

        # 4. 生成compare图（GT和PROJ左右对比）
        if gt_img is not None:
            compare_img = np.hstack([gt_img, proj_img])
            compare_output = compare_dir / f"cam{cam_id}.jpg"
            cv2.imwrite(str(compare_output), compare_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # 5. 生成overlay图（投影叠加到GT上）
        if gt_img is not None:
            overlay_img = gt_img.copy()
            # 找到投影图中非黑色的像素（BGR所有通道都大于阈值）
            mask = np.any(proj_img > 10, axis=2)
            overlay_img[mask] = proj_img[mask]
            overlay_output = overlay_dir / f"cam{cam_id}.jpg"
            cv2.imwrite(str(overlay_output), overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        return True


def main():
    parser = argparse.ArgumentParser(description="基本点云投影 V2 - 点云投影到路侧相机（路侧标定版）")
    parser.add_argument("--roadside-calib", type=str, required=True,
                       help="路侧标定文件路径 (calib.json)")
    parser.add_argument("--roadside-images", type=str, required=True,
                       help="路侧图像文件夹路径")
    parser.add_argument("--camera-id", type=str, required=True,
                       help="标定文件中的camera ID (如 3, 6, 9, 0)")
    parser.add_argument("--pcd", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--timestamp", type=int, required=True)

    args = parser.parse_args()

    projector = UndistortProjectorMultiThread(
        args.roadside_calib, args.roadside_images, args.camera_id
    )
    projector.process_single_frame(
        args.pcd, args.output_dir, args.timestamp
    )

if __name__ == "__main__":
    main()
