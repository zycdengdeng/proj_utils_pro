#!/bin/bash
# 车路协同投影处理系统 - 交互式菜单版
# 支持：单个项目 / 批量项目串行执行 / 自动JSON路径查找

set -e

# 脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示横幅
show_banner() {
    echo -e "${BLUE}"
    echo "======================================================================"
    echo "                  车路协同投影处理系统"
    echo "                    交互式菜单 v2.0"
    echo "======================================================================"
    echo -e "${NC}"
}

# 显示项目菜单
show_project_menu() {
    echo -e "${GREEN}可用的投影项目:${NC}"
    echo "  1) basic        - 基本点云投影"
    echo "  2) blur         - blur投影（路侧相机着色）"
    echo "  3) blur_dense   - blur稠密化投影"
    echo "  4) depth        - depth投影"
    echo "  5) depth_dense  - depth稠密化投影"
    echo "  6) hdmap        - HDMap投影（3D→2D bbox）"
    echo ""
    echo "  7) batch        - 批量处理（选择多个项目串行执行）"
    echo ""
    echo "  0) 退出"
    echo ""
}

# 获取项目信息
get_project_info() {
    local project_type=$1
    local project_dir=""
    local project_name=""

    case $project_type in
        basic|1)
            project_type="basic"
            project_dir="${SCRIPT_DIR}/基本点云投影"
            project_name="基本点云投影"
            ;;
        blur|2)
            project_type="blur"
            project_dir="${SCRIPT_DIR}/blur投影"
            project_name="blur投影（路侧相机着色）"
            ;;
        blur_dense|3)
            project_type="blur_dense"
            project_dir="${SCRIPT_DIR}/blur稠密化投影"
            project_name="blur稠密化投影"
            ;;
        depth|4)
            project_type="depth"
            project_dir="${SCRIPT_DIR}/depth投影"
            project_name="depth投影"
            ;;
        depth_dense|5)
            project_type="depth_dense"
            project_dir="${SCRIPT_DIR}/depth稠密化投影"
            project_name="depth稠密化投影"
            ;;
        hdmap|6)
            project_type="hdmap"
            project_dir="${SCRIPT_DIR}/HDMap投影"
            project_name="HDMap投影（3D→2D bbox）"
            ;;
        *)
            return 1
            ;;
    esac

    echo "$project_type|$project_dir|$project_name"
}

# 检查Python环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: 未找到python3命令${NC}"
        echo "请先安装Python 3"
        exit 1
    fi
}

# 运行单个项目
run_single_project() {
    local project_type=$1
    local batch_script="run_batch_v2.py"

    # 获取项目信息
    local project_info=$(get_project_info "$project_type")
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 未知的项目类型 '$project_type'${NC}"
        return 1
    fi

    IFS='|' read -r project_type project_dir project_name <<< "$project_info"

    echo -e "${BLUE}启动: $project_name${NC}"

    # 检查项目目录
    if [ ! -d "$project_dir" ]; then
        echo -e "${RED}错误: 项目目录不存在: $project_dir${NC}"
        return 1
    fi

    # 检查批处理脚本
    if [ ! -f "$project_dir/$batch_script" ]; then
        echo -e "${RED}错误: 批处理脚本不存在: $project_dir/$batch_script${NC}"
        echo -e "${YELLOW}提示: 该项目可能还未更新到V2版本${NC}"
        return 1
    fi

    # 切换到项目目录并运行
    cd "$project_dir"
    echo -e "${GREEN}项目目录: $project_dir${NC}"
    echo -e "${GREEN}批处理脚本: $batch_script${NC}"
    echo ""

    python3 "$batch_script"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $project_name 处理完成${NC}"
        return 0
    else
        echo -e "${RED}✗ $project_name 处理失败${NC}"
        return 1
    fi
}

# 批量运行多个项目
run_batch_projects() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  批量处理模式${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "可选投影项目："
    echo "  1) basic        - 基本点云投影"
    echo "  2) blur         - blur投影（路侧相机着色）"
    echo "  3) blur_dense   - blur稠密化投影"
    echo "  4) depth        - depth投影"
    echo "  5) depth_dense  - depth稠密化投影"
    echo "  6) hdmap        - HDMap投影（3D→2D bbox）"
    echo ""
    read -p "选择要执行的项目（用逗号或空格分隔，如 1,4,6 或 1 4 6）: " batch_choices

    # 转换逗号为空格
    batch_choices=${batch_choices//,/ }

    if [ -z "$batch_choices" ]; then
        echo -e "${RED}未选择任何项目${NC}"
        return 1
    fi

    # 统计成功/失败
    local total=0
    local success=0
    local failed=0
    local is_first=true

    # 处理每个选择的项目
    for choice in $batch_choices; do
        total=$((total + 1))

        echo ""
        echo -e "${YELLOW}========================================${NC}"
        echo -e "${YELLOW}批量模式: 处理项目 $choice (第 $total 个)${NC}"
        echo -e "${YELLOW}========================================${NC}"

        # 第一个项目：正常交互输入
        # 后续项目：设置环境变量启用批量模式（从配置文件读取）
        if [ "$is_first" = true ]; then
            echo -e "${GREEN}首个项目：输入配置参数...${NC}"
            is_first=false
        else
            echo -e "${GREEN}后续项目：使用已保存的配置...${NC}"
            export PROJECTION_BATCH_MODE=true
        fi

        # 运行项目
        if run_single_project "$choice"; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
            echo -e "${RED}项目 $choice 处理失败，继续下一个...${NC}"
        fi

        echo -e "${YELLOW}========================================${NC}"
    done

    # 清除环境变量
    unset PROJECTION_BATCH_MODE

    # 清除批量配置文件
    rm -f "${SCRIPT_DIR}/temp/projection_batch_config.json"

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  批量处理完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "总数: $total"
    echo -e "成功: ${GREEN}$success${NC}"
    echo -e "失败: ${RED}$failed${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# 主交互循环
interactive_mode() {
    while true; do
        show_banner
        show_project_menu

        read -p "请选择 [1-7, 0]: " choice

        case $choice in
            0)
                echo "退出"
                exit 0
                ;;
            7)
                run_batch_projects
                echo ""
                read -p "按回车继续..."
                ;;
            1|2|3|4|5|6)
                run_single_project "$choice"
                echo ""
                read -p "按回车继续..."
                ;;
            *)
                echo -e "${RED}无效选择${NC}"
                sleep 1
                ;;
        esac
    done
}

# 主程序
main() {
    # 检查Python环境
    check_python

    # 如果有命令行参数，使用命令行模式
    if [ $# -gt 0 ]; then
        if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
            show_banner
            echo "用法: $0 [project_type]"
            echo ""
            echo "命令行模式:"
            echo "  $0 basic         # 运行基本点云投影"
            echo "  $0 depth_dense   # 运行depth稠密化投影"
            echo ""
            echo "交互式模式:"
            echo "  $0               # 不带参数启动交互式菜单"
            echo ""
            exit 0
        fi

        # 命令行模式
        run_single_project "$1"
    else
        # 交互式菜单模式
        interactive_mode
    fi
}

# 执行主程序
main "$@"
