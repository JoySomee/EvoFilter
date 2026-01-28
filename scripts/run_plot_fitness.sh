#!/bin/bash
# =============================================================================
# run_plot_fitness.sh - 最新实验结果自动绘图脚本
# =============================================================================
# 功能: 自动检测results/目录下最新的实验结果，绘制适应度 vs PSNR散点图
#       适合在完成一次实验后快速查看结果分布
#
# 使用方法:
#   bash scripts/run_plot_fitness.sh [每代抽取个体数]
#   示例: bash scripts/run_plot_fitness.sh 20
#         bash scripts/run_plot_fitness.sh      # 使用默认值20
#
# 参数说明:
#   $1 : 每代抽取的个体数量 (默认: 20)
#
# 特性:
#   - 自动检测 results/ 下最新的实验目录
#   - 输出文件带时间戳，避免覆盖
#   - 使用 config.yaml 中的适应度权重配置
#   - 包含最终种群结果 (--include_final)
#
# 输出: 散点图保存在最新实验目录下，文件名格式:
#       fitness_vs_psnr_YYYYMMDD_HHMMSS.png
# =============================================================================

# 绘制最新实验结果的 Fitness vs PSNR 散点图

# 获取最新的结果目录
RESULTS_BASE="results"
LATEST_DIR=$(ls -td "${RESULTS_BASE}"/*/ 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "错误: 未找到结果目录"
    exit 1
fi

echo ">>> 最新结果目录: ${LATEST_DIR}"

# 设置参数
N_SAMPLE_PER_GEN=${1:-20}  # 每代抽取的个体数，默认10
OUTPUT_NAME="fitness_vs_psnr_$(date +%Y%m%d_%H%M%S).png"
OUTPUT_PATH="${LATEST_DIR}/${OUTPUT_NAME}"

# 运行绘图脚本
echo ">>> 开始绘制 Fitness vs PSNR 散点图..."
echo ">>> 每代抽取 ${N_SAMPLE_PER_GEN} 个个体"

python src/plot_fitness_vs_psnr.py \
    --results_dir "${LATEST_DIR}" \
    --n_sample_per_gen ${N_SAMPLE_PER_GEN} \
    --include_final \
    --config config.yaml \
    --output "${OUTPUT_PATH}"

if [ $? -eq 0 ]; then
    echo ">>> 绘图完成!"
    echo ">>> 输出文件: ${OUTPUT_PATH}"
else
    echo ">>> 绘图失败!"
    exit 1
fi
