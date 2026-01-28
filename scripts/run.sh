#!/bin/bash
# =============================================================================
# run.sh - 遗传算法消融实验批量运行脚本
# =============================================================================
# 功能: 运行多组不同适应度权重配置的遗传算法优化实验
#       用于对比不同优化目标（去相关性、信息量、稳定性）对滤光片设计的影响
#
# 使用方法:
#   bash scripts/run.sh
#
# 实验配置:
#   - 实验1: 仅优化去相关性 (w_uncorr=1.0)
#   - 实验2: 仅优化信息量 (w_info=1.0)
#   - 实验3: 仅优化稳定性 (w_stability=1.0) [已注释]
#
# 输出: 结果保存在 results/<timestamp>/ 目录下
# =============================================================================

python src/main.py --generation 50 --w_uncorr 1.0 --w_info 0 --w_stability 0
python src/main.py --generation 50 --w_uncorr 0 --w_info 1.0 --w_stability 0
# python src/main.py --generation 50 --w_uncorr 0 --w_info 0 --w_stability 1.0
