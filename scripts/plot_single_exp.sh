#!/bin/bash
# =============================================================================
# plot_single_exp.sh - 单实验跨代进化轨迹绘图脚本
# =============================================================================
# 功能: 从单个实验的多代种群中抽取个体，绘制适应度 vs PSNR散点图
#       用于分析单次实验中进化过程的收敛趋势
#
# 使用方法:
#   bash scripts/plot_single_exp.sh [结果目录路径] [是否包含最终结果]
#   示例: bash scripts/plot_single_exp.sh "results/2026-01-18 17:59:25" true
#         bash scripts/plot_single_exp.sh  # 使用默认通配符匹配
#
# 参数说明:
#   $1 : 实验结果目录路径 (默认: results/2026-01-14*，自动匹配最新目录)
#   $2 : 是否包含最终种群结果 (默认: true)
#
# 配置变量:
#   N_SAMPLE_PER_GEN : 每代抽取的个体数量 (默认: 20)
#   SEED             : 随机种子，保证可重复性 (默认: 42)
#   OUTPUT           : 输出图片文件名
#
# 适应度权重 (默认配置):
#   w_uncorr=1.0, w_info=0.6, w_stability=0.4
#
# 输出: 散点图保存为 multi_gen_fitness_vs_psnr.png
# =============================================================================

# 从单个实验的多代种群中抽取个体绘制适应度 vs PSNR 图
# 使用方法: ./plot_single_exp.sh [结果目录路径] [是否包含最终结果: true/false]

# 设置默认参数
RESULTS_DIR=${1:-"results/2026-01-14*"}  # 如果没有提供参数，使用通配符匹配最新目录
INCLUDE_FINAL=${2:-"true"}  # 默认包含最终结果
N_SAMPLE_PER_GEN=20  # 每代抽取个体数量
SEED=42      # 随机种子
OUTPUT="multi_gen_fitness_vs_psnr.png"  # 输出文件名

# 如果使用通配符，找到最新的目录
if [[ "$RESULTS_DIR" == *"*"* ]]; then
    RESULTS_DIR=$(ls -td $RESULTS_DIR 2>/dev/null | head -1)
    if [ -z "$RESULTS_DIR" ]; then
        echo "错误: 未找到匹配的结果目录"
        exit 1
    fi
fi

echo "使用结果目录: $RESULTS_DIR"
echo "每代抽取个体数量: $N_SAMPLE_PER_GEN"
echo "包含最终结果: $INCLUDE_FINAL"
echo "输出文件: $OUTPUT"

# 构建命令
CMD="python src/plot_fitness_vs_psnr.py \
    --results_dir \"$RESULTS_DIR\" \
    --n_sample_per_gen $N_SAMPLE_PER_GEN \
    --seed $SEED \
    --output \"$OUTPUT\" \
    --test_data \"../dataset/TSA_simu_data/Truth/\" \
    --w_uncorr 1.0 \
    --w_info 0.6 \
    --w_stability 0.4"

# 如果需要包含最终结果，添加参数
if [ "$INCLUDE_FINAL" = "true" ]; then
    CMD="$CMD --include_final"
fi

# 执行绘图
echo "执行命令: $CMD"
eval $CMD

echo "绘图完成！输出文件: $OUTPUT"