#!/bin/bash
# 统一投影处理启动脚本

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
    echo "======================================================================"
    echo -e "${NC}"
}

# 显示帮助
show_help() {
    show_banner
    echo "用法: $0 <project_type>"
    echo ""
    echo "可用的项目类型:"
    echo "  basic        - 基本点云投影"
    echo "  blur         - blur投影（路侧相机着色）"
    echo "  blur_dense   - blur稠密化投影"
    echo "  depth        - depth投影"
    echo "  depth_dense  - depth稠密化投影"
    echo "  hdmap        - HDMap投影（3D→2D bbox）"
    echo ""
    echo "示例:"
    echo "  $0 basic         # 运行基本点云投影"
    echo "  $0 blur_dense    # 运行blur稠密化投影"
    echo ""
    echo -e "${YELLOW}注意: 请确保已安装Python环境和所需依赖${NC}"
    echo -e "${YELLOW}提示: 使用 ./run_interactive.sh 可获得更多功能（批量处理、自动JSON查找）${NC}"
    echo ""
}

# 检查Python环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: 未找到python3命令${NC}"
        echo "请先安装Python 3"
        exit 1
    fi

    echo -e "${GREEN}✓ Python环境检测通过${NC}"
}

# 运行项目
run_project() {
    local project_type=$1
    local project_dir=""
    local batch_script=""

    case $project_type in
        basic)
            project_dir="${SCRIPT_DIR}/基本点云投影"
            batch_script="run_batch_v2.py"
            echo -e "${BLUE}启动: 基本点云投影${NC}"
            ;;
        blur)
            project_dir="${SCRIPT_DIR}/blur投影"
            batch_script="run_batch_v2.py"
            echo -e "${BLUE}启动: blur投影（路侧相机着色）${NC}"
            ;;
        blur_dense)
            project_dir="${SCRIPT_DIR}/blur稠密化投影"
            batch_script="run_batch_v2.py"
            echo -e "${BLUE}启动: blur稠密化投影${NC}"
            ;;
        depth)
            project_dir="${SCRIPT_DIR}/depth投影"
            batch_script="run_batch_v2.py"
            echo -e "${BLUE}启动: depth投影${NC}"
            ;;
        depth_dense)
            project_dir="${SCRIPT_DIR}/depth稠密化投影"
            batch_script="run_batch_v2.py"
            echo -e "${BLUE}启动: depth稠密化投影${NC}"
            ;;
        hdmap)
            project_dir="${SCRIPT_DIR}/HDMap投影"
            batch_script="run_batch_v2.py"
            echo -e "${BLUE}启动: HDMap投影（3D→2D bbox）${NC}"
            ;;
        *)
            echo -e "${RED}错误: 未知的项目类型 '$project_type'${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac

    # 检查项目目录
    if [ ! -d "$project_dir" ]; then
        echo -e "${RED}错误: 项目目录不存在: $project_dir${NC}"
        exit 1
    fi

    # 检查批处理脚本
    if [ ! -f "$project_dir/$batch_script" ]; then
        echo -e "${RED}错误: 批处理脚本不存在: $project_dir/$batch_script${NC}"
        echo -e "${YELLOW}提示: 该项目可能还未更新到V2版本${NC}"
        exit 1
    fi

    # 切换到项目目录并运行
    cd "$project_dir"
    echo -e "${GREEN}项目目录: $project_dir${NC}"
    echo -e "${GREEN}批处理脚本: $batch_script${NC}"
    echo ""

    python3 "$batch_script"
}

# 主程序
main() {
    show_banner

    # 检查参数
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        show_help
        exit 0
    fi

    # 检查Python环境
    check_python

    # 运行项目
    run_project "$1"
}

# 执行主程序
main "$@"
