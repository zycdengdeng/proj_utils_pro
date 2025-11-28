#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的投影工具模块
提供场景路径查找、变换矩阵加载、批次选择等共享功能
"""

import os
import json
import glob
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tempfile


# ==================== 固定路径配置 ====================
DATASET_ROOT = "/mnt/car_road_data_fix"
SUPPORT_INFO_DIR = os.path.join(DATASET_ROOT, "support_info")

# 车端标定参数目录
VEHICLE_CALIB_DIR = os.path.join(SUPPORT_INFO_DIR, "NoEER705_v3", "camera")

# 路侧标定参数文件
ROADSIDE_CALIB_FILE = os.path.join(SUPPORT_INFO_DIR, "calib.json")

# 自车ID映射文件（support_info目录下）
CARID_JSON_FILE = os.path.join(SUPPORT_INFO_DIR, "carid.json")


# ==================== 场景路径管理 ====================
def find_scene_path(scene_id: str) -> Optional[str]:
    """
    根据场景ID查找完整场景路径

    Args:
        scene_id: 场景ID（如 "002"）

    Returns:
        完整场景路径，如果找不到返回 None
    """
    # 查找所有匹配的目录（只匹配前3位数字）
    pattern = os.path.join(DATASET_ROOT, f"{scene_id}_*")
    matches = glob.glob(pattern)

    if not matches:
        print(f"❌ 未找到场景 {scene_id}")
        return None

    if len(matches) > 1:
        print(f"⚠️  场景 {scene_id} 找到多个匹配目录:")
        for m in matches:
            print(f"   - {m}")
        print(f"使用第一个: {matches[0]}")

    return matches[0]


def get_scene_paths(scene_id: str) -> Dict[str, str]:
    """
    获取场景的所有相关路径

    Args:
        scene_id: 场景ID

    Returns:
        包含各类路径的字典
    """
    scene_root = find_scene_path(scene_id)
    if not scene_root:
        return {}

    paths = {
        'root': scene_root,
        'scene_name': os.path.basename(scene_root),
        'pcd': os.path.join(scene_root, 'road', 'lidar', 'merged_pcd'),
        'roadside_images': os.path.join(scene_root, 'road', 'cameras'),
        'roadside_labels': os.path.join(scene_root, 'road_labels', 'interpolation_labels'),
        'vehicle_images': os.path.join(scene_root, 'car', 'images'),  # 车端GT图像
        'vehicle_calib': VEHICLE_CALIB_DIR,
        'roadside_calib': ROADSIDE_CALIB_FILE
    }

    return paths


def validate_scene_paths(paths: Dict[str, str]) -> bool:
    """
    验证场景路径是否存在

    Args:
        paths: get_scene_paths() 返回的路径字典

    Returns:
        是否所有必需路径都存在
    """
    required = ['pcd', 'vehicle_calib', 'roadside_calib']
    missing = []

    for key in required:
        if not os.path.exists(paths[key]):
            missing.append(f"{key}: {paths[key]}")

    if missing:
        print(f"❌ 缺失路径:")
        for m in missing:
            print(f"   - {m}")
        return False

    return True


# ==================== 变换矩阵管理 ====================
def load_world2lidar_transforms(json_path: str, show_range: bool = False) -> List[Dict]:
    """
    加载 world2lidar 变换矩阵

    Args:
        json_path: 变换JSON文件路径
        show_range: 是否显示时间戳范围

    Returns:
        变换矩阵列表，每个元素包含 timestamp（毫秒）, rotation, translation
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"变换JSON文件不存在: {json_path}")

    with open(json_path, 'r') as f:
        transforms = json.load(f)

    # 时间戳单位转换：检测并转换为毫秒
    if transforms and 'timestamp' in transforms[0]:
        first_ts = transforms[0]['timestamp']

        # 判断时间戳单位：如果小于1e12，认为是秒；否则是毫秒
        if first_ts < 1e12:
            # 秒.微秒格式 → 毫秒
            print(f"   检测到时间戳为秒格式，转换为毫秒...")
            for t in transforms:
                t['timestamp'] = t['timestamp'] * 1000  # 秒 → 毫秒
        else:
            print(f"   检测到时间戳为毫秒格式")

    if show_range and transforms:
        timestamps = [t['timestamp'] for t in transforms]
        print(f"✓ 加载了 {len(transforms)} 个world2lidar变换矩阵")
        print(f"   时间戳范围: {min(timestamps):.0f} ~ {max(timestamps):.0f} (毫秒)")
        print(f"   时间跨度: {(max(timestamps) - min(timestamps)) / 1000:.1f} 秒")
    else:
        print(f"✓ 加载了 {len(transforms)} 个world2lidar变换矩阵")

    return transforms


def find_closest_transform(timestamp_ms: float, transforms: List[Dict],
                          tolerance_ms: float = 5000.0, verbose: bool = False) -> Optional[Dict]:
    """
    根据时间戳查找最接近的变换矩阵

    Args:
        timestamp_ms: 目标时间戳（毫秒）
        transforms: 变换矩阵列表
        tolerance_ms: 容差（毫秒，默认5秒）
        verbose: 是否显示详细匹配信息

    Returns:
        最接近的变换矩阵，如果超出容差返回 None
    """
    min_diff = float('inf')
    closest = None
    closest_ts = None

    for t in transforms:
        diff = abs(t['timestamp'] - timestamp_ms)
        if diff < min_diff:
            min_diff = diff
            closest = t
            closest_ts = t['timestamp']

    if verbose:
        print(f"   时间戳匹配: 目标={timestamp_ms}, 最近={closest_ts}, 差异={min_diff:.1f}ms")

    if min_diff > tolerance_ms:
        if verbose:
            print(f"   ⚠️  差异 {min_diff:.1f}ms 超出容差 {tolerance_ms:.1f}ms")
        return None

    return closest


def transform_points_to_lidar(points_world: np.ndarray, transform: Dict) -> np.ndarray:
    """
    将世界坐标系点云变换到LiDAR坐标系

    Args:
        points_world: 世界坐标系点云 (N, 3)
        transform: world2lidar 变换矩阵字典

    Returns:
        LiDAR坐标系点云 (N, 3)
    """
    # 提取旋转和平移
    rotation = np.array(transform['world2lidar']['rotation'])  # 罗德里格斯向量
    translation = np.array(transform['world2lidar']['translation'])

    # 罗德里格斯向量转旋转矩阵
    import cv2
    R, _ = cv2.Rodrigues(rotation)

    # 应用变换: p_lidar = R @ p_world + t
    points_lidar = (R @ points_world.T).T + translation

    return points_lidar


# ==================== 批次选择逻辑 ====================
def get_batch_files(file_list: List[str], batch_mode: str) -> List[str]:
    """
    根据批次模式选择文件

    Args:
        file_list: 文件列表（已排序）
        batch_mode: 批次模式
            - "all": 所有文件
            - "N": 前N个文件（如 "10"）
            - "middle_90": 中间90个
            - "middle_N": 中间N个（如 "middle_50"）
            - "range_Q_P": 从第Q个开始的P个（如 "range_10_50"）

    Returns:
        选择后的文件列表
    """
    total = len(file_list)

    if batch_mode == "all":
        return file_list

    # 前N个
    if batch_mode.isdigit():
        n = int(batch_mode)
        return file_list[:min(n, total)]

    # 中间90个
    if batch_mode == "middle_90":
        if total < 90:
            print(f"⚠️  总文件数 {total} < 90，返回所有文件")
            return file_list
        start = (total - 90) // 2
        return file_list[start:start+90]

    # 中间N个
    if batch_mode.startswith("middle_"):
        try:
            n = int(batch_mode.split("_")[1])
            if total < n:
                print(f"⚠️  总文件数 {total} < {n}，返回所有文件")
                return file_list
            start = (total - n) // 2
            return file_list[start:start+n]
        except (IndexError, ValueError):
            print(f"❌ 无效的批次模式: {batch_mode}")
            return file_list

    # 从第Q个开始的P个
    if batch_mode.startswith("range_"):
        try:
            parts = batch_mode.split("_")
            q = int(parts[1])  # 起始索引（从1开始）
            p = int(parts[2])  # 数量

            if q < 1:
                print(f"❌ 起始索引必须 >= 1")
                return file_list

            # 转换为0-based索引
            start_idx = q - 1
            end_idx = min(start_idx + p, total)

            if start_idx >= total:
                print(f"❌ 起始索引 {q} 超出范围（总共 {total} 个文件）")
                return []

            return file_list[start_idx:end_idx]
        except (IndexError, ValueError):
            print(f"❌ 无效的批次模式: {batch_mode}，应为 range_Q_P 格式")
            return file_list

    print(f"❌ 未知的批次模式: {batch_mode}")
    return file_list


def print_batch_info(selected_files: List[str], batch_mode: str, original_total: int = None):
    """打印批次选择信息"""
    if original_total is None:
        original_total = len(selected_files)

    selected_count = len(selected_files)
    print(f"\n📊 批次选择: {batch_mode}")
    print(f"   原始文件数: {original_total}")
    print(f"   选择文件数: {selected_count}")

    if batch_mode == "all":
        print(f"   处理范围: 全部")
    elif batch_mode.isdigit():
        n = int(batch_mode)
        print(f"   处理范围: 前 {min(n, original_total)} 个")
    elif batch_mode == "middle_90":
        if original_total >= 90:
            start = (original_total - 90) // 2
            print(f"   处理范围: 索引 [{start+1}, {start+90}]")
        else:
            print(f"   处理范围: 全部（不足90个）")
    elif batch_mode.startswith("middle_"):
        try:
            n = int(batch_mode.split("_")[1])
            if original_total >= n:
                start = (original_total - n) // 2
                print(f"   处理范围: 索引 [{start+1}, {start+n}]")
            else:
                print(f"   处理范围: 全部（不足{n}个）")
        except:
            pass
    elif batch_mode.startswith("range_"):
        try:
            parts = batch_mode.split("_")
            q = int(parts[1])
            p = int(parts[2])
            end = min(q + p - 1, original_total)
            print(f"   处理范围: 索引 [{q}, {end}]（共 {end-q+1} 个）")
        except:
            pass


# ==================== 输出路径管理 ====================
def get_unified_output_paths(output_root: str, scene_id: str, project_type: str) -> Dict[str, str]:
    """
    生成统一的输出路径结构

    Args:
        output_root: 输出根目录
        scene_id: 场景ID
        project_type: 项目类型 (basic, blur, blur_dense, depth, depth_dense, hdmap, json)

    Returns:
        输出路径字典
    """
    scene_output = os.path.join(output_root, scene_id)

    # 统一的输出结构
    paths = {
        'root': scene_output,
        'proj': os.path.join(scene_output, 'proj'),      # 投影结果
        'gt': os.path.join(scene_output, 'gt'),          # 真值图像
        'compare': os.path.join(scene_output, 'compare'),  # GT和PROJ对比图
        'overlay': os.path.join(scene_output, 'overlay'),  # 投影叠加到GT上
    }

    # 根据项目类型添加特定目录
    if project_type in ['depth', 'depth_dense']:
        paths['depth'] = os.path.join(scene_output, 'depth')

    if project_type in ['blur_dense']:
        paths['projections'] = os.path.join(scene_output, 'projections')

    if project_type in ['hdmap', 'json']:
        paths['annotations'] = os.path.join(scene_output, 'annotations')
        paths['videos'] = os.path.join(scene_output, 'videos')

    return paths


def create_output_dirs(paths: Dict[str, str]):
    """创建输出目录"""
    for key, path in paths.items():
        if key == 'root':
            continue
        os.makedirs(path, exist_ok=True)


# ==================== 批量模式配置管理 ====================
# 批量模式配置文件路径（在项目根目录的temp文件夹下）
TEMP_DIR = Path(__file__).resolve().parent / "temp"
BATCH_CONFIG_FILE = TEMP_DIR / "projection_batch_config.json"

def save_batch_config(config: Dict):
    """
    保存批量模式配置到临时文件

    Args:
        config: 配置字典
    """
    # 确保temp目录存在
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    with open(BATCH_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

    # 验证文件确实存在
    if BATCH_CONFIG_FILE.exists():
        print(f"💾 批量配置已保存: {BATCH_CONFIG_FILE}")
    else:
        print(f"⚠️  配置保存失败: {BATCH_CONFIG_FILE}")


def load_batch_config() -> Optional[Dict]:
    """
    从临时文件加载批量模式配置

    Returns:
        配置字典，如果文件不存在返回None
    """
    print(f"🔍 检查配置文件: {BATCH_CONFIG_FILE}")

    if BATCH_CONFIG_FILE.exists():
        print(f"✓ 配置文件存在，加载中...")
        try:
            with open(BATCH_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            print(f"✓ 配置加载成功")
            return config
        except Exception as e:
            print(f"⚠️  配置文件读取失败: {e}")
            return None
    else:
        print(f"✗ 配置文件不存在")
        return None


def clear_batch_config():
    """清除批量模式配置文件"""
    if BATCH_CONFIG_FILE.exists():
        BATCH_CONFIG_FILE.unlink()
        print(f"✓ 已清除批量配置: {BATCH_CONFIG_FILE}")


# ==================== 交互式输入 ====================
def interactive_input(batch_mode_enabled: bool = False) -> Dict:
    """
    统一的交互式输入流程

    Args:
        batch_mode_enabled: 是否启用批量模式（从配置文件读取）

    Returns:
        包含所有输入参数的字典
    """
    # 批量模式：尝试从配置文件加载
    if batch_mode_enabled:
        print(f"\n{'='*60}")
        print("📦 批量模式已启用")
        print(f"{'='*60}")

        config = load_batch_config()
        if config:
            print("\n✓ 使用已保存的配置:")
            print(f"   场景: {', '.join(config['scene_ids'])}")

            # 显示JSON配置（可能是字典或字符串）
            transform_json = config['transform_json']
            if isinstance(transform_json, dict):
                print(f"   JSON: 每个场景使用各自的JSON文件")
                for scene_id, json_path in transform_json.items():
                    print(f"      {scene_id}: {json_path}")
            else:
                print(f"   JSON: {transform_json}")

            print(f"   批次: {config['batch_mode']}")
            print(f"{'='*60}\n")
            return config
        else:
            print("\n⚠️  未找到批量配置文件，切换到交互式输入\n")

    print("\n" + "="*60)
    print("🚀 投影处理系统 - 统一交互界面")
    print("="*60)

    # 1. 输入场景ID（支持多个）
    print("\n📁 步骤 1/3: 输入场景ID")
    print("   提示：可以输入多个场景ID，用空格分隔")
    print("   示例：002 004 005")
    scene_input = input("   请输入场景ID: ").strip()
    scene_ids = scene_input.split()

    if not scene_ids:
        print("❌ 未输入场景ID")
        return None

    print(f"   ✓ 选择了 {len(scene_ids)} 个场景: {', '.join(scene_ids)}")

    # 验证场景路径
    valid_scenes = []
    for sid in scene_ids:
        paths = get_scene_paths(sid)
        if paths and validate_scene_paths(paths):
            valid_scenes.append(sid)
            print(f"   ✓ 场景 {sid}: {paths['scene_name']}")
        else:
            print(f"   ✗ 场景 {sid}: 路径无效，跳过")

    if not valid_scenes:
        print("❌ 没有有效的场景")
        return None

    # 2. 自动查找或手动输入world2lidar变换JSON路径
    print("\n🔄 步骤 2/3: 选择world2lidar变换JSON路径")
    print("   选项：")
    print("     - auto    : 自动查找（从 transform_json/{场景ID}/world2lidar_transforms.json）")
    print("     - manual  : 手动输入路径")
    json_mode = input("   请选择模式 [auto]: ").strip() or "auto"

    if json_mode == "auto":
        # 自动查找模式
        print("\n   使用自动查找模式...")
        json_base_dir = Path(__file__).resolve().parent / "transform_json"

        # 为每个场景查找对应的JSON文件
        transform_jsons = {}
        all_valid = True

        for scene_id in valid_scenes:
            json_path = json_base_dir / scene_id / "world2lidar_transforms.json"

            if json_path.exists():
                transform_jsons[scene_id] = str(json_path)
                print(f"   ✓ 场景 {scene_id}: {json_path}")
            else:
                print(f"   ✗ 场景 {scene_id}: 未找到 JSON 文件 {json_path}")
                all_valid = False

        if not all_valid:
            print("\n   ❌ 部分场景缺少 JSON 文件")
            use_manual = input("   是否切换到手动模式? (y/n) [y]: ").strip().lower() or 'y'
            if use_manual != 'y':
                return None
            json_mode = "manual"
        else:
            # 检查是否所有场景可以使用同一个JSON（路径相同）
            unique_jsons = list(set(transform_jsons.values()))
            if len(unique_jsons) == 1:
                # 所有场景使用同一个JSON
                transform_json = unique_jsons[0]
                print(f"\n   ✓ 所有场景使用相同的变换文件: {transform_json}")
            else:
                # 多个不同的JSON，每个场景使用自己的JSON
                print(f"\n   ✓ 每个场景使用各自的变换文件:")
                for sid in valid_scenes:
                    print(f"      场景 {sid}: {transform_jsons[sid]}")
                # 保存映射关系而不是单个路径
                transform_json = transform_jsons

    if json_mode == "manual":
        # 手动输入模式
        print("\n   手动输入模式...")
        print("   示例：/mnt/car_road_data_fix/transforms/world2lidar.json")
        transform_json = input("   请输入路径: ").strip()

        if not os.path.exists(transform_json):
            print(f"❌ 文件不存在: {transform_json}")
            return None

        print(f"   ✓ 变换文件: {transform_json}")

    # 3. 选择批次模式
    print("\n📊 步骤 3/3: 选择批次模式")
    print("   选项：")
    print("     - all          : 处理所有文件（默认）")
    print("     - N            : 处理前N个（例如：10）")
    print("     - middle_90    : 处理中间90个")
    print("     - middle_N     : 处理中间N个（例如：middle_50）")
    print("     - range_Q_P    : 从第Q个开始处理P个（例如：range_10_50）")
    batch_mode = input("   请输入批次模式 [all]: ").strip() or "all"

    print(f"   ✓ 批次模式: {batch_mode}")

    # 返回配置（不包含并行配置，由各项目单独处理）
    config = {
        'scene_ids': valid_scenes,
        'transform_json': transform_json,
        'batch_mode': batch_mode
    }

    print("\n" + "="*60)
    print("✓ 配置完成")
    print("="*60 + "\n")

    # 保存基础配置（后续会追加并行配置等）
    save_batch_config(config)

    return config


def get_parallel_config(batch_mode_enabled: bool = False) -> Dict:
    """
    获取并行配置（进程数和线程数）

    Args:
        batch_mode_enabled: 是否启用批量模式

    Returns:
        包含 num_processes 和 threads_per_frame 的字典
    """
    import multiprocessing as mp

    # 批量模式：从配置文件读取
    if batch_mode_enabled:
        config = load_batch_config()
        if config and 'num_processes' in config and 'threads_per_frame' in config:
            print("\n⚙️  多核并行设置:")
            print(f"   使用已保存的配置: {config['num_processes']}进程 × {config['threads_per_frame']}线程")
            return {
                'num_processes': config['num_processes'],
                'threads_per_frame': config['threads_per_frame']
            }

    # 交互式输入
    print("\n⚙️  多核并行设置:")
    total_cores = mp.cpu_count()
    print(f"   检测到 {total_cores} 个CPU核心")

    # 推荐配置
    if total_cores >= 128:
        default_processes = 16
    elif total_cores >= 64:
        default_processes = 8
    elif total_cores >= 32:
        default_processes = 4
    else:
        default_processes = 2

    default_threads = 7

    print(f"   推荐配置: {default_processes}进程 × {default_threads}线程")

    num_processes = input(f"   并行进程数 [默认{default_processes}]: ").strip()
    num_processes = int(num_processes) if num_processes else default_processes

    threads_per_frame = input(f"   每帧线程数 [默认{default_threads}]: ").strip()
    threads_per_frame = int(threads_per_frame) if threads_per_frame else default_threads

    print(f"\n   ✓ 配置: {num_processes}进程 × {threads_per_frame}线程")

    # 更新配置文件（追加并行配置）
    config = load_batch_config()
    if config:
        config['num_processes'] = num_processes
        config['threads_per_frame'] = threads_per_frame
        save_batch_config(config)

    return {
        'num_processes': num_processes,
        'threads_per_frame': threads_per_frame
    }


def get_ego_vehicle_id(scene_ids: List[str], batch_mode_enabled: bool = False,
                       default_id: int = 45) -> Dict[str, int]:
    """
    获取自车ID配置（用于HDMap投影）

    支持三种模式：
    1. 自动模式（auto）：从carid.json自动加载每个场景的自车ID
    2. 单一ID模式：所有场景使用相同的自车ID
    3. 批量模式：从配置文件读取

    Args:
        scene_ids: 场景ID列表
        batch_mode_enabled: 是否启用批量模式
        default_id: 默认自车ID

    Returns:
        场景ID→自车ID的映射字典，如 {"002": 29, "003": 45}
    """
    # 批量模式：从配置文件读取
    if batch_mode_enabled:
        config = load_batch_config()
        if config and 'ego_vehicle_mapping' in config:
            print(f"\n🚗 自车配置:")
            print(f"   使用已保存的配置:")
            ego_mapping = config['ego_vehicle_mapping']
            for sid in scene_ids:
                ego_id = ego_mapping.get(sid, default_id)
                print(f"      场景 {sid}: 自车ID = {ego_id}")
            return ego_mapping

    # 交互式输入
    print(f"\n🚗 自车ID配置:")
    print(f"   选项：")
    print(f"     - auto    : 自动从 carid.json 加载（推荐）")
    print(f"     - <数字>  : 所有场景使用相同ID（如 45）")

    mode = input(f"   请选择模式 [auto]: ").strip() or "auto"

    ego_mapping = {}

    if mode == "auto":
        # 自动模式：从carid.json加载
        print(f"\n   使用自动模式，从 carid.json 加载...")
        carid_mapping = load_carid_mapping()

        if not carid_mapping:
            print(f"   ❌ 未能加载carid.json，切换到手动模式")
            mode = str(default_id)
        else:
            # 为每个场景查找对应的自车ID
            all_valid = True
            for scene_id in scene_ids:
                if scene_id in carid_mapping:
                    ego_mapping[scene_id] = carid_mapping[scene_id]
                    print(f"   ✓ 场景 {scene_id}: 自车ID = {carid_mapping[scene_id]}")
                else:
                    print(f"   ✗ 场景 {scene_id}: 未找到自车ID，使用默认值 {default_id}")
                    ego_mapping[scene_id] = default_id
                    all_valid = False

            if not all_valid:
                confirm = input(f"\n   部分场景使用默认ID，是否继续? (y/n) [y]: ").strip().lower() or 'y'
                if confirm != 'y':
                    print(f"   取消配置")
                    return {}

    if mode != "auto":
        # 单一ID模式：所有场景使用相同的ID
        try:
            ego_id = int(mode)
            print(f"\n   所有场景使用相同的自车ID: {ego_id}")
            for scene_id in scene_ids:
                ego_mapping[scene_id] = ego_id
                print(f"   ✓ 场景 {scene_id}: 自车ID = {ego_id}")
        except ValueError:
            print(f"   ❌ 无效输入，使用默认值 {default_id}")
            for scene_id in scene_ids:
                ego_mapping[scene_id] = default_id

    # 保存到配置文件
    config = load_batch_config()
    if config:
        config['ego_vehicle_mapping'] = ego_mapping
        save_batch_config(config)

    return ego_mapping


# ==================== 自车ID映射管理 ====================
def load_carid_mapping(carid_json_path: Optional[str] = None) -> Dict[str, int]:
    """
    从 carid.json 加载场景ID→自车ID映射

    Args:
        carid_json_path: carid.json文件路径，默认使用根目录下的文件

    Returns:
        场景ID到自车ID的映射字典，如 {"002": 29, "003": 45, ...}
    """
    if carid_json_path is None:
        carid_json_path = CARID_JSON_FILE

    if not os.path.exists(carid_json_path):
        print(f"⚠️  未找到carid.json文件: {carid_json_path}")
        return {}

    try:
        with open(carid_json_path, 'r') as f:
            data = json.load(f)

        mapping = {}
        for item in data.get('results', []):
            clip_name = item.get('clip_name', '')
            nearest_carid = item.get('nearest_carid')

            # 从clip_name提取场景ID（前3位数字）
            # 例如 "002_car0325_road0327_t2" → "002"
            if clip_name and nearest_carid is not None:
                scene_id = clip_name.split('_')[0]
                mapping[scene_id] = nearest_carid

        print(f"✓ 加载了 {len(mapping)} 个场景的自车ID映射")
        return mapping

    except Exception as e:
        print(f"❌ 加载carid.json失败: {e}")
        return {}


# ==================== 工具函数 ====================
def extract_timestamp_from_filename(filename: str) -> Optional[float]:
    """
    从文件名提取时间戳

    支持格式：
        - 123456789.pcd
        - merged_123456789.pcd
        - cam0_123456789.png
    """
    import re
    basename = os.path.basename(filename)
    # 匹配数字（可能带小数点）
    match = re.search(r'(\d+)(?:\.\d+)?\.', basename)
    if match:
        return float(match.group(1))
    return None


def sort_files_by_timestamp(files: List[str]) -> List[str]:
    """按时间戳排序文件列表"""
    files_with_ts = []
    for f in files:
        ts = extract_timestamp_from_filename(f)
        if ts is not None:
            files_with_ts.append((ts, f))

    # 按时间戳排序
    files_with_ts.sort(key=lambda x: x[0])
    return [f for _, f in files_with_ts]


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("测试 common_utils.py")
    print("="*60)

    # 测试场景路径查找
    print("\n1. 测试场景路径查找:")
    paths = get_scene_paths("002")
    if paths:
        for key, value in paths.items():
            print(f"   {key}: {value}")

    # 测试批次选择
    print("\n2. 测试批次选择:")
    test_files = [f"file_{i:03d}.txt" for i in range(100)]

    for mode in ["all", "10", "middle_90", "middle_50", "range_10_20"]:
        selected = get_batch_files(test_files, mode)
        print(f"   模式 '{mode}': 选择了 {len(selected)} 个文件")
        if len(selected) <= 5:
            print(f"     文件: {selected}")
        else:
            print(f"     首尾文件: {selected[0]}, ..., {selected[-1]}")

    print("\n✓ 测试完成")
