import numpy as np
from cal_response import cal_response_batch
import scipy.io as scio


class FitnessEvaluator:
    def __init__(self, spectral_library='database_spe.mat'):
        self.spectral_library = np.array(scio.loadmat(spectral_library)['spe'])
        self.bands = 301
        # 如果没有提供光谱库，生成一个随机模拟库
        if spectral_library is None:
            print("Warning: 未提供真实光谱库，使用随机高斯峰模拟库。")
            self.Cov_s = self._generate_mock_covariance(self.bands)
        else:
            # 计算先验协方差矩阵 C_s = S.T @ S
            # S 形状: (Samples, W)
            self.Cov_s = self.spectral_library.T @ self.spectral_library

    def _generate_mock_covariance(self, dim):
        """生成模拟的光谱协方差矩阵 (用于测试)"""
        # 模拟 100 个具有不同中心波长的随机高斯光谱
        num_samples = 100
        S = np.zeros((dim, num_samples))
        x = np.linspace(0, 1, dim)
        for i in range(num_samples):
            center = np.random.rand()
            width = np.random.uniform(0.05, 0.1)
            S[:, i] = np.exp(-(x - center)**2 / (2 * width**2))
        return S @ S.T

    def calculate_uncorrelation_score(self, F):
        """
        计算指标 1: 滤光片之间的不相关性 (Uncorrelation)
        方法: 1 - (相关系数矩阵非对角元素的平均绝对值)
        """
        # F 形状: (16, W)
        # 计算行与行之间的相关系数矩阵 (16, 16)
        corr_matrix = np.corrcoef(F)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        # 取绝对值
        abs_corr = np.abs(corr_matrix)
        
        # 将对角线元素 (即自相关，都是1) 设为0，只计算互相关
        np.fill_diagonal(abs_corr, 0)

        # 计算平均互相关系数
        # 矩阵大小是 W x W，非对角元素个数为 W * (W - 1)
        num_wavelengths = F.shape[0]
        # 计算平均互相关系数
        num_off_diagonal = num_wavelengths * (num_wavelengths - 1)
        if num_off_diagonal > 0:
            mean_cross_corr = np.sum(abs_corr) / num_off_diagonal
        else:
            mean_cross_corr = 0.0        
        # 分数越高越好 (0=完全相关, 1=完全不相关)
        score = 1.0 - mean_cross_corr
        return score, abs_corr

    def calculate_information_score(self, F):
        """
        计算指标 2: 降维性能 / 信息保持度 (Information Preservation)
        方法: D-Optimality -> log(det(F @ Cov_s @ F.T))
        """
        # 1. 计算测量协方差矩阵 C_y (16, 16)
        # C_y = F * C_s * F_T
        C_y = F @ self.Cov_s @ F.T
        
        # 2. 计算行列式 log(det)
        # 增加一个极小的噪声 eye * epsilon 防止矩阵奇异导致 log(0) 报错
        epsilon = 1e-10
        C_y_stable = C_y + np.eye(C_y.shape[0]) * epsilon
        
        # 使用 sign, logdet 计算以保持数值稳定性
        sign, logdet = np.linalg.slogdet(C_y_stable)
        
        if sign <= 0:
            return -np.inf # 矩阵奇异或非正定，性能极差
        
        return logdet
    
    def calculate_stability_score(self, G):
        """
        计算指标 3: 数值稳定性 (Numerical Stability)
        物理意义: 衡量逆问题的病态程度 (Condition Number)。
        如果此值过高，任何微小的传感器噪声都会在重建时被无限放大。
        """
        try:
            # 计算条件数 cond(A) = ||A|| * ||A^-1||
            cond_number = np.linalg.cond(G)
        except np.linalg.LinAlgError:
            cond_number = 1e6 # 如果矩阵奇异，给一个巨大的惩罚值
            
        # 为了方便优化，Loss 通常取对数，因为条件数可能从 10 变化到 10^16
        log_cond = np.log10(cond_number) if cond_number > 0 else 0
        # 1. 计算 Loss (越小越好): 0 (Best) -> 1 (Worst)
        loss_stability = np.clip(log_cond / 10.0, 0.0, 1.0)
        stability_score = 1.0 - loss_stability
        
        return cond_number, stability_score

    def evaluate(self, individual_filters):
        """
        主评估函数
        """
        # 1. 物理仿真: 得到光谱矩阵 F (16, W)
        F = cal_response_batch(individual_filters)
        # 2. 计算指标
        score_uncorr, corr_matrix = self.calculate_uncorrelation_score(F)
        score_info = self.calculate_information_score(F)
        _, stability_score = self.calculate_stability_score(F)
        
        return {
            "spectra": F,
            "score_uncorrelation": score_uncorr, # (0~1), 越大越好
            "score_information": score_info,     # (log scale), 越大越好
            "stability_score": stability_score, # (0~1), 越大越好
            "correlation_matrix": corr_matrix    # 用于debug
        }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from initial import FilterInitializer
    # 1. 准备波长范围 (400nm - 700nm)
    wavelengths = np.linspace(400, 700, 301)
    initializer = FilterInitializer()
    # Generate a 2D array of thickness designs
    thick_designs, labels = initializer.generate_individual_4x4()
    # 2. 初始化仿真器和评估器
    evaluator = FitnessEvaluator() # 使用模拟光谱库
    print("正在计算适应度...")
    results = evaluator.evaluate(thick_designs)
    # 5. 输出结果
    print("-" * 30)
    print(f"【指标1】列不相关性得分 (0-1): {results['score_uncorrelation']:.4f}")
    print(f"       (平均波长间相关系数: {1 - results['score_uncorrelation']:.4f})")
    print(f"       (这代表了系统的光谱分辨潜力)")
    print(f"【指标2】信息保持得分 (LogDet): {results['score_information']:.4f}")
    print(f"       (这代表了系统的光谱降维能力)")
    print(f"【指标3】数值稳定性得分 (0-1): {results['stability_score']:.4f}")
    print(f"       (这代表了系统的数值计算稳定性)")
    print("-" * 30)
    
    # 6. 可视化结果
    plt.figure(figsize=(12, 5))
    # 图1: 16条光谱曲线
    plt.subplot(1, 2, 1)
    plt.plot(wavelengths, results['spectra'].T, alpha=0.6, linewidth=1)
    plt.title("Spectral Response of 16 Filters")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmittance")
    plt.grid(linestyle=':')
    # 图2: 列相关性矩阵热力图 (WxW)
    # 注意：这可能是一个很大的矩阵 (如 401x401)
    plt.subplot(1, 2, 2)
    im = plt.imshow(results['correlation_matrix'], cmap='coolwarm', vmin=0, vmax=1, origin='lower')
    plt.colorbar(im, label="Correlation Coefficient")
    plt.title(f"Wavelength Column Correlation\n(Matrix Size: {16}x{16})")
    plt.xlabel("Wavelength Index")
    plt.ylabel("Wavelength Index")
    
    plt.tight_layout()
    plt.savefig('filter_spectra_correlation.png', dpi=300)