import os
import numpy as np
import scipy.io as scio
from scipy.interpolate import CubicSpline
from tqdm import tqdm

def sample(img, threshold=0.5, n_samples=150):
    l1_norm = np.sum(np.abs(img), axis=2)  # shape: (H, W)
    valid_mask = l1_norm > threshold
    valid_coords = np.argwhere(valid_mask)

    if valid_coords.shape[0] < n_samples:
        return None  # 表示采样失败

    # 随机无放回采样
    selected_indices = np.random.choice(valid_coords.shape[0], size=n_samples, replace=False)
    selected_points = valid_coords[selected_indices]
    spectra = img[selected_points[:, 0], selected_points[:, 1], :]  # (n_samples, C)
    # 对光谱进行插值，从31个点到301个点
    original_wavelengths = np.arange(400, 701, 10)
    target_wavelengths = np.arange(400, 701, 1)
    
    # 使用三次样条插值 (Cubic Spline Interpolation)
    interpolated_spectra = np.apply_along_axis(
        lambda spectrum: CubicSpline(original_wavelengths, spectrum)(target_wavelengths),
        axis=1,
        arr=spectra
    )
    
    return interpolated_spectra


if __name__ == '__main__':
    img_dir = '../dataset/CAVE/'
    output_path = './database_spe.mat'
    filelist = [f for f in os.listdir(img_dir) if f.endswith('.mat')]  # 只处理 .mat 文件
    all_spectra = []  # 动态收集
    np.random.seed(42)
    for i, file in enumerate(tqdm(filelist)):
        filepath = os.path.join(img_dir, file)
        try:
            data = scio.loadmat(filepath)['hsi']  # 假设变量名为 'hsi'
        except Exception as e:
            print(f"⚠️ 跳过文件 {file}：加载失败 - {e}")
            continue
        spectra = sample(data)
        if spectra is None:
            print(f"⚠️ 跳过文件 {file}：有效像素不足 150 个")
            continue
        all_spectra.append(spectra)
    if not all_spectra:
        raise RuntimeError("没有成功采样的图像！")
    spe_db = np.vstack(all_spectra)  # shape: (N * 150, 31)，其中 N 是成功采样的图像数
    print(f"成功采样 {len(all_spectra)} 张图像，总光谱数: {spe_db.shape[0]}")
    scio.savemat(output_path, {'spe': spe_db})