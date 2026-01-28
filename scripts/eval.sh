#!/bin/bash
# =============================================================================
# eval.sh - 高光谱图像重建评估批量脚本
# =============================================================================
# 功能: 使用不同的滤光片mask对高光谱图像进行重建，并计算PSNR/SSIM指标
#       用于对比不同适应度等级滤光片的重建性能
#
# 使用方法:
#   bash scripts/eval.sh
#
# 评估的mask文件 (按适应度从高到低):
#   - mask_best_28_cond.mat   : 最优适应度滤光片
#   - mask_high_28_cond.mat   : 高适应度滤光片
#   - mask_mid_28_cond.mat    : 中等适应度滤光片
#   - mask_second_28_cond.mat : 次优适应度滤光片
#   - mask_low_28_cond.mat    : 低适应度滤光片
#
# 参数说明:
#   --test_data  : 测试数据路径（高光谱真值图像）
#   --mask       : 滤光片mask文件路径
#   --output     : 重建结果输出目录
#   --lambda_reg : FISTA正则化参数 (默认0.005)
#   --max_iter   : FISTA最大迭代次数 (默认100)
#
# 输出: 重建结果保存在 recon_results/ 目录下
# =============================================================================

python src/evaluate.py \
    --test_data ../dataset/TSA_simu_data/Truth/ \
    --mask mask/ablation/mask_best_28_cond.mat \
    --output recon_results \
    --lambda_reg 0.005 \
    --max_iter 100

python src/evaluate.py \
    --test_data ../dataset/TSA_simu_data/Truth/ \
    --mask mask/ablation/mask_high_28_cond.mat \
    --output recon_results \
    --lambda_reg 0.005 \
    --max_iter 100

python src/evaluate.py \
    --test_data ../dataset/TSA_simu_data/Truth/ \
    --mask mask/ablation/mask_mid_28_cond.mat \
    --output recon_results \
    --lambda_reg 0.005 \
    --max_iter 100

python src/evaluate.py \
    --test_data ../dataset/TSA_simu_data/Truth/ \
    --mask mask/ablation/mask_second_28_cond.mat \
    --output recon_results \
    --lambda_reg 0.005 \
    --max_iter 100

python src/evaluate.py \
    --test_data ../dataset/TSA_simu_data/Truth/ \
    --mask mask/ablation/mask_low_28_cond.mat \
    --output recon_results \
    --lambda_reg 0.005 \
    --max_iter 100
