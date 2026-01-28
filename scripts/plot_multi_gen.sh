#!/bin/bash
# =============================================================================
# plot_multi_gen.sh - 多实验跨代进化轨迹绘图脚本
# =============================================================================
# 功能: 从多个不同配置的实验中，每代随机抽取个体绘制适应度 vs PSNR图
#       用于对比不同适应度函数（去相关性、信息量、条件数）的进化效果
#
# 使用方法:
#   bash scripts/plot_multi_gen.sh [是否包含最终结果: true/false]
#   示例: bash scripts/plot_multi_gen.sh true
#
# 参数说明:
#   $1 : 是否包含最终种群结果 (默认: true)
#
# 配置变量:
#   UNCORR_RESULTS_DIR : 去相关性实验结果目录
#   INFO_RESULTS_DIR   : 信息量实验结果目录
#   COND_RESULTS_DIR   : 条件数实验结果目录
#   N_SAMPLE_PER_GEN   : 每代抽取的个体数量 (默认: 20)
#   SEED               : 随机种子，保证可重复性
#
# 输出文件:
#   - plots/multi_gen_uncorr_vs_psnr.png : 去相关性进化轨迹
#   - plots/multi_gen_info_vs_psnr.png   : 信息量进化轨迹
#   - plots/multi_gen_cond_vs_psnr.png   : 条件数进化轨迹
# =============================================================================

# 从多代种群中每代独立抽取个体绘制适应度 vs PSNR 图
# 使用方法: ./plot_multi_gen.sh [是否包含最终结果: true/false]

# 设置默认参数
UNCORR_RESULTS_DIR="results/2026-01-15 16:20:10" # Uncorrelation实验结果目录
INFO_RESULTS_DIR="results/2026-01-14 17:39:45" # Information实验结果目录  
COND_RESULTS_DIR="results/2026-01-14 17:40:11" # Condition实验结果目录

INCLUDE_FINAL=${1:-"true"}  # 默认包含最终结果
N_SAMPLE_PER_GEN=20  # 每代抽取个体数量
SEED=42      # 随机种子

UNCORR_OUTPUT="plots/multi_gen_uncorr_vs_psnr.png"  # 输出文件名
INFO_OUTPUT="plots/multi_gen_info_vs_psnr.png"  # 输出文件名
COND_OUTPUT="plots/multi_gen_cond_vs_psnr.png"  # 输出文件名

echo "包含最终结果: $INCLUDE_FINAL"
echo "每代抽取个体数量: $N_SAMPLE_PER_GEN"

# 构建基础命令
build_cmd() {
    local results_dir=$1
    local output=$2
    local w_uncorr=$3
    local w_info=$4
    local w_stability=$5
    
    local cmd="python src/plot_fitness_vs_psnr.py \
        --results_dir \"$results_dir\" \
        --n_sample_per_gen $N_SAMPLE_PER_GEN \
        --seed $SEED \
        --output \"$output\" \
        --test_data \"../dataset/TSA_simu_data/Truth/\" \
        --w_uncorr $w_uncorr \
        --w_info $w_info \
        --w_stability $w_stability"
    
    # 如果需要包含最终结果，添加参数
    if [ "$INCLUDE_FINAL" = "true" ]; then
        cmd="$cmd --include_final"
    fi
    
    echo "$cmd"
}

# 构建各个命令
UNCORR_CMD=$(build_cmd "$UNCORR_RESULTS_DIR" "$UNCORR_OUTPUT" 1.0 0 0)
INFO_CMD=$(build_cmd "$INFO_RESULTS_DIR" "$INFO_OUTPUT" 0 1.0 0)
COND_CMD=$(build_cmd "$COND_RESULTS_DIR" "$COND_OUTPUT" 0 0 1.0)

# 执行绘图
# echo "=== 执行 Uncorrelation 实验绘图 ==="
# echo "使用目录: $UNCORR_RESULTS_DIR"
# eval $UNCORR_CMD

echo ""
echo "=== 执行 Information 实验绘图 ==="
echo "使用目录: $INFO_RESULTS_DIR"
eval $INFO_CMD

echo ""
echo "=== 执行 Condition 实验绘图 ==="
echo "使用目录: $COND_RESULTS_DIR"
eval $COND_CMD

echo ""
echo "所有绘图完成！"
echo "输出文件:"
echo "  - $UNCORR_OUTPUT"
echo "  - $INFO_OUTPUT"  
echo "  - $COND_OUTPUT"