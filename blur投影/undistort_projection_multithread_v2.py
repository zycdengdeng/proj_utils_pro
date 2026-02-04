#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去畸变版投影：多线程CPU优化版 V2 - blur投影（路侧标定版 lableRoadside）
新管道：加载PCD → 从所有4个路侧相机着色 → 投影到单个路侧相机（通过camera_id）
使用virtualLidarToCam变换：点云(≈VirtualLidar) → R_V2C @ points + t_V2C → camera → image
"""

import json
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import argparse
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import re

# 添加父目录到路径以导入 common_utils（使用绝对路径）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common_utils

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
    """找到路侧相机图像（用于着色）
    Args:
        max_time_diff_ms: 最大允许的时间差（毫秒），默认1000ms
    """
    camera_folder = Path(roadside_images_folder) / pinhole_name
    if not camera_folder.exists():
        return None, None

    # 图像格式: cam{cam_id}_{timestamp}.png
    expected_name = f"cam{cam_id}_{timestamp_ms}.png"
    img_path = camera_folder / expected_name

    if img_path.exists():
        return img_path, 0

    # 如果精确匹配失败，尝试找最接近的时间戳
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

    # 只返回时间差在允许范围内的图像
    if closest_file and min_diff <= max_time_diff_ms:
        return closest_file, min_diff

    return None, min_diff if closest_file else None


def find_roadside_gt_image(roadside_images_folder, cam_id, timestamp_ms, max_time_diff_ms=1000):
    """找到路侧相机GT图像（用于投影对比）
    Args:
        roadside_images_folder: 路侧图像根目录
        cam_id: camera ID（如 "3", "6", "9", "0"）
        timestamp_ms: 时间戳（毫秒）
        max_time_diff_ms: 最大允许的时间差（毫秒）
    Returns:
        (img_path, time_diff) 或 (None, None)
    """
    # cam_id到pinhole的映射
    cam_id_to_pinhole = {
        "3": "pinhole0",
        "6": "pinhole1",
        "9": "pinhole2",
        "0": "pinhole3"
    }

    pinhole_name = cam_id_to_pinhole.get(str(cam_id))
    if not pinhole_name:
        return None, None

    camera_folder = Path(roadside_images_folder) / pinhole_name
    if not camera_folder.exists():
        return None, None

    # 查找cam{cam_id}_{timestamp}.png
    expected_name = f"cam{cam_id}_{timestamp_ms}.png"
    img_path = camera_folder / expected_name

    if img_path.exists():
        return img_path, 0

    # 如果精确匹配失败，尝试找最接近的时间戳
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

    # 只返回时间差在允许范围内的图像
    if closest_file and min_diff <= max_time_diff_ms:
        return closest_file, min_diff

    return None, min_diff if closest_file else None


class BlurProjectorMultiThread:
    def __init__(self, roadside_calib_path, roadside_images_folder, camera_id):
        """
        初始化投影器

        Args:
            roadside_calib_path: 路侧标定文件路径 (calib.json)
            roadside_images_folder: 路侧图像文件夹路径
            camera_id: 目标投影相机ID（字符串，如 "3", "6", "9", "0"）
        """
        # 加载路侧标定文件
        with open(roadside_calib_path, 'r') as f:
            self.roadside_calib = json.load(f)

        self.roadside_images_folder = Path(roadside_images_folder)
        self.camera_id = str(camera_id)

        # 验证camera_id是否存在
        if self.camera_id not in self.roadside_calib["camera"]:
            available_ids = list(self.roadside_calib["camera"].keys())
            raise ValueError(
                f"Camera ID '{self.camera_id}' 不存在于标定文件中。"
                f"可用的ID: {available_ids}"
            )

        # 存储相机参数
        self.roadside_camera_params = {}  # 所有4个相机（用于着色）
        self.target_camera_params = None   # 目标相机（用于投影）

        # 设置OpenCV线程数
        cv2.setNumThreads(0)

        # 加载所有路侧相机参数（用于着色）
        for pinhole_id in range(4):
            self.load_roadside_camera_params(pinhole_id)

        # 加载目标相机参数（用于投影）
        self.load_target_camera_params()

        print(f"✓ 投影器初始化完成: 目标相机ID = {self.camera_id}")

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

    def load_target_camera_params(self):
        """加载目标投影相机参数"""
        cam_config = self.roadside_calib["camera"][self.camera_id]

        K = np.asarray(cam_config["intri"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(cam_config.get("distor", []), dtype=np.float64).reshape(-1) if "distor" in cam_config else None
        is_fisheye = bool(cam_config.get("isFish", 0))

        R_V2C = rodrigues_to_R(cam_config["virtualLidarToCam"]["rotate"])
        t_V2C = np.asarray(cam_config["virtualLidarToCam"]["trans"], dtype=np.float64).reshape(3, 1)

        if is_fisheye:
            resolution = tuple(self.roadside_calib["imgSize"]["fish"])
        else:
            resolution = tuple(self.roadside_calib["imgSize"]["notFish"])

        self.target_camera_params = {
            'K': K,
            'D': D,
            'R_V2C': R_V2C,
            't_V2C': t_V2C,
            'is_fisheye': is_fisheye,
            'resolution': resolution
        }

    def colorize_pointcloud_from_roadside(self, points, timestamp_ms):
        """使用路侧相机给点云着色（保持原有实现）
        Args:
            points: 点云坐标 (N, 3) - 在VirtualLidar坐标系
            timestamp_ms: 时间戳
        Returns:
            colors: 点云颜色 (N, 3) - RGB值在[0,1]范围
        """
        N = len(points)
        colors = np.zeros((N, 3), dtype=np.float32)
        color_counts = np.zeros(N, dtype=np.int32)

        # 遍历每个路侧相机
        for pinhole_id in range(4):
            pinhole_name = ROADSIDE_CAMERAS[pinhole_id]['name']
            cam_id = ROADSIDE_CAMERAS[pinhole_id]['cam_id']

            # 查找图像
            img_path, time_diff = find_roadside_image(
                self.roadside_images_folder, pinhole_name, cam_id, timestamp_ms
            )
            if not img_path:
                print(f"  警告: 未找到{pinhole_name}的图像 (查找cam{cam_id}_{timestamp_ms}.png)")
                continue

            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  警告: 无法读取{img_path}")
                continue

            if time_diff > 0:
                print(f"  {pinhole_name}: 使用图像 {img_path.name} (时间差: {time_diff}ms)")

            # 获取相机参数
            params = self.roadside_camera_params[pinhole_id]
            K = params['K']
            D = params['D']
            R_V2C = params['R_V2C']
            t_V2C = params['t_V2C']
            is_fisheye = params['is_fisheye']
            img_w, img_h = params['resolution']

            # VirtualLidar → 相机坐标系
            points_cam = (R_V2C @ points.T).T + t_V2C.T

            # 过滤背后的点
            valid_mask = points_cam[:, 2] > 0.1
            if not np.any(valid_mask):
                continue

            valid_indices = np.where(valid_mask)[0]
            points_valid = points_cam[valid_mask]

            # 投影到图像平面
            if D is not None and len(D) > 0:
                # 使用畸变模型投影
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

            # 过滤图像内的点
            valid_proj_mask = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & \
                             (uv[:, 1] >= 0) & (uv[:, 1] < img_h)

            if not np.any(valid_proj_mask):
                continue

            valid_proj_indices = valid_indices[valid_proj_mask]
            uv_valid = uv[valid_proj_mask].astype(int)

            # 从图像中提取颜色
            for i, (u, v) in enumerate(uv_valid):
                point_idx = valid_proj_indices[i]
                # OpenCV图像是BGR，转为RGB
                bgr = img[v, u]
                rgb = bgr[::-1] / 255.0  # BGR->RGB, 归一化到[0,1]
                colors[point_idx] += rgb
                color_counts[point_idx] += 1

            print(f"  {pinhole_name}: 着色了 {len(uv_valid)} 个点")

        # 平均多个相机的颜色
        colored_mask = color_counts > 0
        colors[colored_mask] /= color_counts[colored_mask, np.newaxis]

        # 未着色的点使用灰色
        uncolored_count = np.sum(~colored_mask)
        if uncolored_count > 0:
            colors[~colored_mask] = 0.5
            print(f"  警告: {uncolored_count} 个点未被路侧相机着色（使用灰色）")

        colored_count = np.sum(colored_mask)
        print(f"  总计: {colored_count}/{N} 个点成功着色 ({colored_count/N*100:.1f}%)")

        return colors

    def project_to_target_camera(self, points, colors):
        """
        投影到目标路侧相机

        变换流程：点云(VirtualLidar坐标系) → R_V2C @ points + t_V2C → camera → image

        Args:
            points: 点云坐标 (N, 3) - VirtualLidar坐标系
            colors: 点云颜色 (N, 3) - RGB在[0,1]范围

        Returns:
            img: 投影图像
            count: 投影点数量
        """
        params = self.target_camera_params
        K = params['K']
        D = params['D']
        R_V2C = params['R_V2C']
        t_V2C = params['t_V2C']
        is_fisheye = params['is_fisheye']
        img_w, img_h = params['resolution']

        # VirtualLidar → 相机坐标系
        points_cam = (R_V2C @ points.T).T + t_V2C.T

        # 过滤背后的点
        valid = points_cam[:, 2] > 0.1
        if not np.any(valid):
            return np.zeros((img_h, img_w, 3), dtype=np.uint8), 0

        points_valid = points_cam[valid]
        colors_valid = colors[valid] if colors is not None else None

        # 投影到图像平面
        if D is not None and len(D) > 0:
            # 使用畸变模型投影
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

        if len(uv_valid) > 0:
            proj_colors = (colors_valid[valid_proj] * 255).astype(np.uint8)
            # 使用cv2.circle绘制半径为2的圆
            for (u, v), color in zip(uv_valid, proj_colors):
                cv2.circle(img, (u, v), 2,
                         (int(color[2]), int(color[1]), int(color[0])), -1)

        return img, len(uv_valid)

    def process_single_frame(self, pcd_path, output_dir, timestamp_ms):
        """
        处理单帧数据

        Args:
            pcd_path: PCD文件路径
            output_dir: 输出目录
            timestamp_ms: 时间戳（毫秒）

        Returns:
            是否成功
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

        if len(points) == 0:
            print(f"❌ 点云为空: {pcd_path}")
            return False

        print(f"✓ 加载点云: {len(points)} 个点")

        # 2. 使用路侧相机给点云着色
        print(f"🎨 使用路侧相机着色点云...")
        colors = self.colorize_pointcloud_from_roadside(points, timestamp_ms)

        # 3. 投影到目标相机
        print(f"📸 投影到目标相机 (ID={self.camera_id})...")
        proj_img, count = self.project_to_target_camera(points, colors)

        # 保存投影结果
        proj_output = proj_dir / f"cam{self.camera_id}.jpg"
        cv2.imwrite(str(proj_output), proj_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"  投影了 {count} 个点")

        # 4. 处理GT图像
        gt_img_path, time_diff = find_roadside_gt_image(
            self.roadside_images_folder, self.camera_id, timestamp_ms
        )

        if gt_img_path:
            gt_img = cv2.imread(str(gt_img_path))
            if gt_img is not None:
                gt_output = gt_dir / f"cam{self.camera_id}.jpg"
                cv2.imwrite(str(gt_output), gt_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

                if time_diff > 0:
                    print(f"  GT图像: {gt_img_path.name} (时间差: {time_diff}ms)")

                # 生成compare图（GT和PROJ左右对比）
                compare_img = np.hstack([gt_img, proj_img])
                compare_output = compare_dir / f"cam{self.camera_id}.jpg"
                cv2.imwrite(str(compare_output), compare_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

                # 生成overlay图（投影叠加到GT上）
                overlay_img = gt_img.copy()
                mask = np.any(proj_img > 10, axis=2)
                overlay_img[mask] = proj_img[mask]
                overlay_output = overlay_dir / f"cam{self.camera_id}.jpg"
                cv2.imwrite(str(overlay_output), overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            else:
                print(f"  警告: 无法读取GT图像 {gt_img_path}")
        else:
            print(f"  警告: 未找到GT图像 (查找cam{self.camera_id}_{timestamp_ms}.png)")

        print(f"✓ 帧处理完成: {timestamp_ms}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Blur投影 V2 - 路侧标定版")
    parser.add_argument("--roadside-calib", type=str, required=True,
                       help="路侧标定文件路径 (calib.json)")
    parser.add_argument("--roadside-images", type=str, required=True,
                       help="路侧图像文件夹路径")
    parser.add_argument("--camera-id", type=str, required=True,
                       help="目标投影相机ID (如 3, 6, 9, 0)")
    parser.add_argument("--pcd", type=str, required=True,
                       help="PCD文件路径")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--timestamp", type=int, required=True,
                       help="时间戳（毫秒）")

    args = parser.parse_args()

    # 创建投影器
    projector = BlurProjectorMultiThread(
        args.roadside_calib,
        args.roadside_images,
        args.camera_id
    )

    # 处理单帧
    success = projector.process_single_frame(
        args.pcd,
        args.output_dir,
        args.timestamp
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
