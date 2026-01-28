import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import scipy.io as sio  # 用于加载 .mat 文件
import os
from scipy.ndimage import convolve
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
import time
import argparse

# We find that this calculation method is more close to DGSMP's.
def numpy_psnr(img, ref):  # input [28,256,256]
    img = np.round(img * 256)
    ref = np.round(ref * 256)
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = np.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * np.log10((255 * 255) / mse)
    return psnr / nC

def _gaussian_window(window_size, sigma):
    """生成1D高斯窗口"""
    x = np.arange(window_size)
    gauss = np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
    return gauss / gauss.sum()

def _create_window_2d(window_size, sigma=1.5):
    """生成2D高斯窗口"""
    gauss_1d = _gaussian_window(window_size, sigma)
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    return gauss_2d

def numpy_ssim(img, ref, window_size=11):  # input [28,256,256]
    """
    计算SSIM，与torch版本保持一致
    img, ref: shape [C, H, W]
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    window = _create_window_2d(window_size)
    
    nC = img.shape[0]
    ssim_val = 0
    
    for i in range(nC):
        img_ch = img[i, :, :]
        ref_ch = ref[i, :, :]
        
        # 使用卷积计算局部均值
        mu1 = convolve(img_ch, window, mode='constant')
        mu2 = convolve(ref_ch, window, mode='constant')
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = convolve(img_ch ** 2, window, mode='constant') - mu1_sq
        sigma2_sq = convolve(ref_ch ** 2, window, mode='constant') - mu2_sq
        sigma12 = convolve(img_ch * ref_ch, window, mode='constant') - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        ssim_val += ssim_map.mean()
    
    return ssim_val / nC

# ==========================================
# 第一部分：数据加载 (加载真实高光谱文件)
# ==========================================

def load_real_data(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        test_data[i, :, :, :] = img
    return test_data

def generate_masks(mask_path):             # for optical filters-based system
    mask = sio.loadmat(mask_path)
    mask_3d = mask['mask']
    return mask_3d
# ==========================================
# 第二部分：模拟成像系统 (Mosaic Pattern)
# ==========================================

def generate_filters_and_measurement(hsi_data, filters, pattern_size=4):
    """
    模拟快照式光谱相机的成像过程：
    1. 生成滤光片传输矩阵 A
    2. 应用马赛克阵列 (Mosaic) 生成 RAW 图像
    """
    print(">>> [Step 2] Simulating Measurement Process...")
    H, W, n_bands = hsi_data.shape
    flatten_hsi = hsi_data.reshape(-1, n_bands)
    flatten_mask = filters.reshape(-1, n_bands)
    raw_image = flatten_hsi @ flatten_mask.T
    return raw_image

# ==========================================
# 第三部分：凸优化求解器 (FISTA)
# ==========================================

def fista_solver_batch(Y, A, Psi, lambda_reg=0.01, max_iter=50):
    """
    批量 FISTA 求解器
    Y: (N_samples, M) -> (HW, 16)
    A: (M, N) -> (16, 28)
    Psi: (N, N) -> (28, 28)
    """
    # 1. 构建等效字典 D = A * Psi -> (M, N)
    D = A @ Psi
    
    # 2. 预计算常数
    L = np.linalg.norm(D, ord=2) ** 2
    step_size = 1.0 / (L + 1e-8)
    
    Dt = D.T # (N, M)
    
    # 预计算 D^T * Y (矩阵乘法)
    # (N, M) @ (N_samples, M).T -> (N, N_samples)
    # 为了方便计算，我们全程保持 (N_samples, Features) 的形状，所以 transpose 一下
    Dt_Y = (Dt @ Y.T).T # -> (N_samples, N)
    
    Dt_D = D.T @ D # (N, N)
    
    # 初始化
    N_samples = Y.shape[0]
    n_coeffs = Psi.shape[1]
    
    theta = np.zeros((N_samples, n_coeffs))
    y_aux = np.zeros((N_samples, n_coeffs))
    t = 1.0
    
    # 迭代求解
    for k in range(max_iter):
        theta_old = theta.copy()
        
        # 梯度计算: grad = (D^T * D * theta - D^T * Y)
        # 这里的矩阵乘法方向: (N_samples, N) @ (N, N) -> (N_samples, N)
        term1 = y_aux @ Dt_D.T # 注意 transpose，因为 y_aux 是行向量形式
        grad = term1 - Dt_Y
        
        z = y_aux - step_size * grad
        
        # 软阈值
        threshold = lambda_reg * step_size
        theta = np.sign(z) * np.maximum(np.abs(z) - threshold, 0)
        
        # 动量
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y_aux = theta + ((t - 1) / t_new) * (theta - theta_old)
        t = t_new
        
    # 重建 X = theta * Psi^T (因为 theta 是行向量形式)
    # (N_samples, N) @ (N, N) -> (N_samples, N)
    x_hat = theta @ Psi.T
    
    return np.maximum(x_hat, 0)

def reconstruction_pipeline(Y_flat, A_matrix):
    """
    重建流水线
    """
    print(f">>> [Step 3] Starting Reconstruction (Solving Y = X * A^T)...")
    
    M, N = A_matrix.shape
    
    # 准备稀疏基
    Psi = idct(np.eye(N), norm='ortho', axis=0)
    
    start_time = time.time()
    
    # 批量求解
    # 直接将所有像素作为一个巨大的 Batch 传入
    # 如果内存不足，FISTA 内部会自动利用 Numpy 的广播机制，通常这一步很快
    print(f"    Solving for {Y_flat.shape[0]} pixels...")
    X_recon_flat = fista_solver_batch(Y_flat, A_matrix, Psi, lambda_reg=0.005, max_iter=100)
    
    print(f"    Reconstruction finished in {time.time() - start_time:.2f}s")
    
    return X_recon_flat

# ==========================================
# 主程序运行
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description='高光谱图像重建与评估')
    parser.add_argument('--test_data', '-t', type=str, 
                        default='../dataset/TSA_simu_data/Truth/',
                        help='测试数据路径 (包含.mat文件的目录)')
    parser.add_argument('--mask', '-m', type=str,
                        default='mask/mask.mat',
                        help='滤光片/掩膜文件路径 (.mat格式)')
    parser.add_argument('--output', '-o', type=str, 
                        default='recon_results',
                        help='输出结果保存目录')
    parser.add_argument('--lambda_reg', '-l', type=float, 
                        default=0.005,
                        help='FISTA正则化参数 (默认: 0.005)')
    parser.add_argument('--max_iter', '-i', type=int, 
                        default=100,
                        help='FISTA最大迭代次数 (默认: 100)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    test_data_path = args.test_data
    mask_path = args.mask
    output_dir = args.output

    try:
        # 1. 加载真实数据
        batch_hsi_data = load_real_data(test_data_path)
        mask3d = generate_masks(mask_path)
        n_image, H, W, n_bands = batch_hsi_data.shape
        print(">>> [Step 5] Saving Results...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 准备日志文件
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(output_dir, f"eval_results_{timestamp}.txt")
        log_lines = []
        log_lines.append(f"Evaluation Results - {timestamp}")
        log_lines.append("=" * 50)
        log_lines.append(f"Test Data: {test_data_path}")
        log_lines.append(f"Mask: {mask_path}")
        log_lines.append(f"Lambda: {args.lambda_reg}, Max Iter: {args.max_iter}")
        log_lines.append("=" * 50)
        
        all_psnr = []
        all_ssim = []
        for i in range(n_image):
            raw_image = generate_filters_and_measurement(batch_hsi_data[i], mask3d)
            x_recon_flat = reconstruction_pipeline(raw_image, mask3d.reshape(-1, n_bands))
            x_recon = x_recon_flat.reshape(H, W, n_bands)
            
            # 转换为 [C, H, W] 格式计算指标
            pred_chw = x_recon.transpose(2, 0, 1)  # [28, 256, 256]
            truth_chw = batch_hsi_data[i].transpose(2, 0, 1)  # [28, 256, 256]
            
            psnr_val = numpy_psnr(pred_chw, truth_chw)
            ssim_val = numpy_ssim(pred_chw, truth_chw)
            rmse = np.sqrt(np.mean((batch_hsi_data[i] - x_recon) ** 2))
            
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            
            result_line = f"Image {i}: PSNR={psnr_val:.4f} dB, SSIM={ssim_val:.4f}, RMSE={rmse:.6f}"
            print(f"    {result_line}")
            log_lines.append(result_line)
            
            mat_save_path = os.path.join(output_dir, f"reconstructed_cube_{i}.mat")
            sio.savemat(mat_save_path, {
                "pred": x_recon,
                "truth": batch_hsi_data[i],
                "metrics": {"rmse": rmse, "psnr": psnr_val, "ssim": ssim_val}
            })
            print(f"    Saved .mat data to: {mat_save_path}")
        
        # 汇总结果
        log_lines.append("=" * 50)
        log_lines.append(f"Mean PSNR: {np.mean(all_psnr):.4f} dB")
        log_lines.append(f"Mean SSIM: {np.mean(all_ssim):.4f}")
        
        print("\n>>> [Summary] Average Metrics:")
        print(f"    Mean PSNR: {np.mean(all_psnr):.4f} dB")
        print(f"    Mean SSIM: {np.mean(all_ssim):.4f}")
        
        # 保存日志文件
        with open(log_path, 'w') as f:
            f.write('\n'.join(log_lines))
        print(f"\n>>> Results saved to: {log_path}")
        
    except Exception as e:
        print(f"Error: {e}")