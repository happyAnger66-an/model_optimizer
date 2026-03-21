#!/usr/bin/env bash
# Model Optimizer 安装脚本
# 用法: bash scripts/install.sh [base|all|pi05|dev]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQUIREMENTS_DIR="$PROJECT_ROOT/requirements"

# 默认安装模式
MODE="${1:-all}"

# 可选: 国内镜像
PIP_INDEX="${PIP_INDEX:-}"
if [[ -n "$PIP_INDEX" ]]; then
    PIP_EXTRA="-i $PIP_INDEX"
else
    PIP_EXTRA=""
fi

echo "=========================================="
echo "Model Optimizer 安装"
echo "模式: $MODE"
echo "项目根目录: $PROJECT_ROOT"
echo "=========================================="

cd "$PROJECT_ROOT"

case "$MODE" in
    base)
        echo "[1/2] 安装基础依赖..."
        pip install -r "$REQUIREMENTS_DIR/requirements-base.txt" $PIP_EXTRA
        echo "[2/2] 以开发模式安装 model_optimizer..."
        pip install -e . $PIP_EXTRA
        ;;
    all)
        echo "[1/2] 安装完整依赖..."
        pip install -r "$REQUIREMENTS_DIR/requirements-base.txt" $PIP_EXTRA
        echo "[2/2] 以开发模式安装 model_optimizer (all extras)..."
        pip install -e ".[all]" $PIP_EXTRA
        ;;
    pi05)
        echo "[1/3] 安装基础依赖..."
        pip install -r "$REQUIREMENTS_DIR/requirements-base.txt" $PIP_EXTRA
        echo "[2/3] 安装 π0.5 依赖..."
        pip install -r "$REQUIREMENTS_DIR/requirements-pi05.txt" $PIP_EXTRA
        echo "[3/3] 以开发模式安装 model_optimizer (all + pi05)..."
        pip install -e ".[all,pi05]" $PIP_EXTRA
        echo ""
        echo "注意: π0.5 模型需额外配置 openpi 源码路径至 PYTHONPATH"
        ;;
    dev)
        echo "[1/2] 安装开发依赖..."
        pip install -r "$REQUIREMENTS_DIR/requirements-dev.txt" $PIP_EXTRA
        echo "[2/2] 以开发模式安装 model_optimizer..."
        pip install -e ".[all,dev-test]" $PIP_EXTRA
        ;;
    *)
        echo "用法: $0 [base|all|pi05|dev]"
        echo ""
        echo "  base  - 仅核心依赖 + 开发模式安装"
        echo "  all   - 完整功能(量化/导出/YOLO/WebUI)"
        echo "  pi05  - all + π0.5 模型依赖"
        echo "  dev   - 开发与测试"
        exit 1
        ;;
esac

echo ""
echo "安装完成. 验证: model-optimizer-cli version"
