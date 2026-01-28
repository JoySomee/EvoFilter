#!/bin/bash
# =============================================================================
# plot.sh - 适应度与PSNR关系图绘制脚本（指定种群文件）
# =============================================================================
# 功能: 从指定的种群文件中绘制适应度(Fitness) vs PSNR散点图
#       用于分析不同适应度函数配置下，适应度与重建质量的相关性
#
# 使用方法:
#   bash scripts/plot.sh
#
# 绘制的实验对比:
#   - uncorr_vs_psnr.png  : 去相关性适应度 vs PSNR
#   - stable_vs_psnr.png  : 稳定性适应度 vs PSNR
#   - info_vs_psnr.png    : 信息量适应度 vs PSNR
#
# 参数说明:
#   -p : 种群文件路径 (.npy格式)
#   -t : 测试数据路径
#   -o : 输出图片文件名
#   --w_uncorr/w_info/w_stability : 适应度权重配置
#
# 输出: 散点图保存在当前目录
# =============================================================================

python src/plot_fitness_vs_psnr.py -p results/2026-01-13\ 16:26:49/final_population.npy -t ../dataset/TSA_simu_data/Truth/ -o uncorr_vs_psnr.png --w_uncorr 1.0 --w_info 0 --w_stability 0
python src/plot_fitness_vs_psnr.py -p results/2026-01-13\ 16:34:27/final_population.npy -t ../dataset/TSA_simu_data/Truth/ -o stable_vs_psnr.png --w_uncorr 0 --w_info 0 --w_stability 1.0
python src/plot_fitness_vs_psnr.py -p results/2026-01-13\ 18:15:33/final_population.npy -t ../dataset/TSA_simu_data/Truth/ -o info_vs_psnr.png --w_uncorr 0 --w_info 1.0 --w_stability 0
