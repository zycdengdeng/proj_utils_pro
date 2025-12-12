#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense Projection Video Maker

从投影输出结果生成视频
支持 depth、depth_dense、blur、blur_dense、basic 等投影类型
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# 相机名称
CAMERA_NAMES = ['FN', 'FW', 'FL', 'FR', 'RL', 'RR', 'RN']


def get_sorted_timestamp_folders(scene_dir):
    """获取场景目录下所有时间戳文件夹，按时间戳排序"""
    scene_path = Path(scene_dir)

    if not scene_path.exists():
        print(f"警告: 场景目录不存在: {scene_dir}")
        return []

    timestamp_folders = []
    for folder in scene_path.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            timestamp_folders.append(folder)

    timestamp_folders.sort(key=lambda x: int(x.name))
    return timestamp_folders


def create_video_from_images(image_paths, output_path, fps, target_resolution=None):
    """
    从图像列表创建视频

    Args:
        image_paths: 图像路径列表
        output_path: 输出视频路径
        fps: 帧率
        target_resolution: 目标分辨率 (width, height)，None则使用原始分辨率
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        print(f"警告: 没有图像可用于创建视频")
        return False

    # 读取第一张图像获取分辨率
    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        print(f"警告: 无法读取第一张图像 {image_paths[0]}")
        return False

    if target_resolution is None:
        target_resolution = (first_img.shape[1], first_img.shape[0])

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        target_resolution
    )

    # 写入帧
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue

        # 缩放到目标分辨率
        if img.shape[1] != target_resolution[0] or img.shape[0] != target_resolution[1]:
            img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)

        video_writer.write(img)

    video_writer.release()
    return True


def process_clip(clip_dir, output_dir, fps, subdirs, cameras, target_resolution=None):
    """
    处理单个clip，生成视频

    Args:
        clip_dir: clip输出目录（包含时间戳子目录）
        output_dir: 视频输出目录
        fps: 帧率
        subdirs: 要处理的子目录列表（如 ['proj', 'gt', 'compare', 'overlay', 'depth']）
        cameras: 相机列表
        target_resolution: 目标分辨率
    """
    clip_path = Path(clip_dir)
    output_path = Path(output_dir)
    clip_name = clip_path.name

    print(f"\n{'='*60}")
    print(f"处理 Clip: {clip_name}")
    print(f"{'='*60}")

    # 获取所有时间戳文件夹
    timestamp_folders = get_sorted_timestamp_folders(clip_path)

    if not timestamp_folders:
        print(f"跳过: 未找到时间戳文件夹")
        return False

    print(f"找到 {len(timestamp_folders)} 帧")
    print(f"时间戳范围: {timestamp_folders[0].name} ~ {timestamp_folders[-1].name}")

    # 处理每个子目录
    for subdir in subdirs:
        print(f"\n处理子目录: {subdir}")

        # 检查子目录是否存在
        first_subdir = timestamp_folders[0] / subdir
        if not first_subdir.exists():
            print(f"  跳过: 子目录不存在 {first_subdir}")
            continue

        # 处理每个相机
        for cam_name in cameras:
            # 收集图像路径
            image_paths = []
            for folder in timestamp_folders:
                # 尝试 .jpg 和 .png
                for ext in ['.jpg', '.png']:
                    img_path = folder / subdir / f"{cam_name}{ext}"
                    if img_path.exists():
                        image_paths.append(img_path)
                        break

            if not image_paths:
                print(f"  {cam_name}: 无图像")
                continue

            # 生成视频
            video_filename = f"{clip_name}_{subdir}_{cam_name}.mp4"
            video_output = output_path / subdir / video_filename

            if create_video_from_images(image_paths, video_output, fps, target_resolution):
                print(f"  {cam_name}: {len(image_paths)} 帧 -> {video_output}")
            else:
                print(f"  {cam_name}: 生成失败")

    return True


def main():
    parser = argparse.ArgumentParser(description='Dense Projection Video Maker')

    parser.add_argument('--input-dir', type=str, required=True,
                        help='投影输出目录（包含时间戳子目录）')

    parser.add_argument('--output-dir', type=str, required=True,
                        help='视频输出目录')

    parser.add_argument('--fps', type=int, default=10,
                        help='视频帧率（默认 10 FPS）')

    parser.add_argument('--subdirs', type=str, nargs='+',
                        default=['proj', 'gt', 'compare', 'overlay'],
                        help='要处理的子目录（默认: proj gt compare overlay）')

    parser.add_argument('--cameras', type=str, nargs='+',
                        default=CAMERA_NAMES,
                        help='相机列表（默认: 全部7个相机）')

    parser.add_argument('--resolution', type=str, default=None,
                        help='目标分辨率，如 1280x720（默认: 保持原始分辨率）')

    args = parser.parse_args()

    # 解析分辨率
    target_resolution = None
    if args.resolution:
        try:
            w, h = args.resolution.lower().split('x')
            target_resolution = (int(w), int(h))
        except:
            print(f"警告: 无法解析分辨率 {args.resolution}，使用原始分辨率")

    print("="*60)
    print("Dense Projection Video Maker")
    print("="*60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"帧率: {args.fps} FPS")
    print(f"子目录: {args.subdirs}")
    print(f"相机: {args.cameras}")
    print(f"分辨率: {target_resolution or '原始'}")

    # 处理
    process_clip(
        args.input_dir,
        args.output_dir,
        args.fps,
        args.subdirs,
        args.cameras,
        target_resolution
    )

    print(f"\n{'='*60}")
    print(f"完成! 视频保存到: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
