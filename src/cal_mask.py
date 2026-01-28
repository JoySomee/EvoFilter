import numpy as np
from cal_response import cal_response_batch
import scipy.io as scio


bands = [453.3, 457.6, 462.1, 466.8, 471.6, 476.5, 481.6, 
         486.9, 492.4, 498.0, 503.9, 509.9, 516.2, 522.7, 
         529.5, 536.5, 543.8, 551.4, 558.6, 567.5, 575.3, 
         584.3, 594.4, 604.2, 614.4, 625.1, 636.3, 648.1]
         
thick_dir = 'results/2026-01-13 05:48:18/best_individual_93.npy'
thick_list = np.load(thick_dir)
mask_dir = 'mask/ablation/mask_low_28_cond.mat'
T_lists = cal_response_batch(thick_list, bands, 'data/nk_map_special.pkl')
mask = T_lists.reshape(4, 4, 28)
#large_mask = np.tile(mask, (64, 64, 1))
scio.savemat(mask_dir, {'mask': mask})

