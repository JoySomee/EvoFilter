import numpy as np
from cal_response import cal_response_batch
import scipy.io as scio


class FitnessEvaluator:
    def __init__(self, spectral_library='data/database_spe.mat', corr_mode='row'):
        """
        初始化适应度评估器

        Args:
            spectral_library: 光谱库路径
            corr_mode: 不相关性计算模式
                - 'row': 行不相关性，计算滤光片之间的相关性 (16x16)
                - 'column': 列不相关性，计算波长之间的相关性 (WxW)
                - 'both': 同时计算行和列，取较差的分数
        """
        self.corr_mode = corr_mode
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

    def calculate_uncorrelation_score(self, F, mode=None):
        """
        计算不相关性得分

        Args:
            F: 光谱响应矩阵，形状 (16, W)，16个滤光片，W个波长点
            mode: 计算模式，覆盖实例默认值
                - 'row': 行不相关性 - 计算滤光片之间的相关性
                - 'column': 列不相关性 - 计算波长之间的相关性
                - 'both': 同时计算，返回较差的分数

        Returns:
            score: 不相关性得分 (0~1)，越大越好
            corr_matrix: 相关系数矩阵 (用于可视化)

        物理意义:
            - row模式: 衡量不同滤光片光谱响应之间的独立性
              如果两个滤光片响应高度相关，说明信息冗余
            - column模式: 衡量不同波长采样之间的独立性
              如果两个波长的响应高度相关，说明波长分辨率有限
        """
        mode = mode or self.corr_mode

        if mode == 'row':
            return self._calc_row_uncorrelation(F)
        elif mode == 'column':
            return self._calc_column_uncorrelation(F)
        elif mode == 'both':
            row_score, row_matrix = self._calc_row_uncorrelation(F)
            col_score, col_matrix = self._calc_column_uncorrelation(F)
            # 返回较差的分数 (木桶效应)
            if row_score <= col_score:
                return row_score, row_matrix
            else:
                return col_score, col_matrix
        else:
            raise ValueError(f"未知的相关性计算模式: {mode}. 可选: 'row', 'column', 'both'")

    def _calc_row_uncorrelation(self, F):
        """
        行不相关性: 计算滤光片之间的相关性

        F 形状: (16, W) -> 相关矩阵 (16, 16)
        衡量不同滤光片的光谱响应是否独立
        """
        # np.corrcoef(F) 计算行与行之间的相关系数矩阵
        corr_matrix = np.corrcoef(F)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        abs_corr = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr, 0)

        max_cross_corr = np.max(abs_corr)
        score = 1.0 - max_cross_corr

        return score, abs_corr

    def _calc_column_uncorrelation(self, F):
        """
        列不相关性: 计算波长之间的相关性

        F 形状: (16, W) -> F.T 形状: (W, 16) -> 相关矩阵 (W, W)
        衡量不同波长的滤光片响应是否独立
        """
        # 转置后计算列与列之间的相关系数
        corr_matrix = np.corrcoef(F.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        abs_corr = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr, 0)

        max_cross_corr = np.max(abs_corr)
        score = 1.0 - max_cross_corr

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

    def calculate_transmittance_score(self, F, min_transmittance=0.5, max_std=0.15):
        """
        计算指标 4: 透过率质量 (Transmittance Quality)

        物理意义:
        1. 平均透过率过低会导致信噪比下降，传感器接收的光子数不足
        2. 各通道透过率差异过大会导致动态范围不均匀，部分通道信息丢失

        参数:
            F: 光谱响应矩阵 (16, W)
            min_transmittance: 最低可接受的平均透过率阈值 (默认 0.5)
            max_std: 最大可接受的通道间标准差 (默认 0.15)

        返回:
            score: 综合得分 (0~1), 越大越好
            details: 包含各项统计量的字典
        """
        # F 形状: (16, W), 16个滤光片，W个波长点
        num_filters = F.shape[0]

        # 1. 计算每个滤光片的平均透过率
        channel_mean_trans = np.mean(F, axis=1)  # shape: (16,)

        # 2. 计算所有通道的整体平均透过率
        overall_mean_trans = np.mean(channel_mean_trans)

        # 3. 计算通道间透过率的标准差 (衡量均匀性)
        channel_std_trans = np.std(channel_mean_trans)

        # 4. 计算透过率均值得分
        # 如果 overall_mean >= min_transmittance, 得满分
        # 否则按比例惩罚
        if overall_mean_trans >= min_transmittance:
            mean_score = 1.0
        else:
            # 线性惩罚: 0 -> 0分, min_transmittance -> 1分
            mean_score = overall_mean_trans / min_transmittance

        # 5. 计算均匀性得分
        # 如果 channel_std <= max_std, 得满分
        # 否则按比例惩罚
        if channel_std_trans <= max_std:
            uniformity_score = 1.0
        else:
            # 超过 max_std 的部分进行惩罚
            # std = max_std -> 1分, std = 2*max_std -> 0.5分, std越大分越低
            uniformity_score = max_std / channel_std_trans

        # 6. 综合得分: 两者加权平均
        # 均值更重要 (权重 0.6), 均匀性次之 (权重 0.4)
        score = 0.6 * mean_score + 0.4 * uniformity_score

        details = {
            'channel_mean_transmittance': channel_mean_trans,  # 每个通道的平均透过率
            'overall_mean_transmittance': overall_mean_trans,  # 整体平均透过率
            'channel_std_transmittance': channel_std_trans,    # 通道间标准差
            'mean_score': mean_score,                          # 均值子得分
            'uniformity_score': uniformity_score               # 均匀性子得分
        }

        return score, details

    def evaluate(self, individual_filters):
        """
        主评估函数
        """
        # 1. 物理仿真: 得到光谱矩阵 F (16, W)
        F = cal_response_batch(individual_filters)
        # 2. 计算指标
        score_uncorr, corr_matrix = self.calculate_uncorrelation_score(F)
        score_info = self.calculate_information_score(F)
        cond_number, stability_score = self.calculate_stability_score(F)
        transmittance_score, trans_details = self.calculate_transmittance_score(F)

        return {
            "spectra": F,
            "score_uncorrelation": score_uncorr,        # (0~1), 越大越好
            "score_information": score_info,            # (log scale), 越大越好
            "stability_score": stability_score,         # (0~1), 越大越好
            "condition_number": cond_number,            # 条件数原始值
            "transmittance_score": transmittance_score, # (0~1), 越大越好
            "transmittance_details": trans_details,     # 透过率详细信息
            "correlation_matrix": corr_matrix           # 用于debug
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
    print(f"【指标1】滤光片不相关性得分 (0-1): {results['score_uncorrelation']:.4f}")
    print(f"       (最大滤光片间相关系数: {1 - results['score_uncorrelation']:.4f})")
    print(f"       (这代表了滤光片响应的独立性)")
    print(f"【指标2】信息保持得分 (LogDet): {results['score_information']:.4f}")
    print(f"       (这代表了系统的光谱降维能力)")
    print(f"【指标3】数值稳定性得分 (0-1): {results['stability_score']:.4f}")
    print(f"       (这代表了系统的数值计算稳定性)")
    trans_details = results['transmittance_details']
    print(f"【指标4】透过率质量得分 (0-1): {results['transmittance_score']:.4f}")
    print(f"       整体平均透过率: {trans_details['overall_mean_transmittance']:.4f}")
    print(f"       通道间标准差: {trans_details['channel_std_transmittance']:.4f}")
    print(f"       均值子得分: {trans_details['mean_score']:.4f}, 均匀性子得分: {trans_details['uniformity_score']:.4f}")
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
    # 图2: 滤光片相关性矩阵热力图 (16x16)
    plt.subplot(1, 2, 2)
    im = plt.imshow(results['correlation_matrix'], cmap='coolwarm', vmin=0, vmax=1, origin='lower')
    plt.colorbar(im, label="Correlation Coefficient")
    plt.title(f"Filter Correlation Matrix\n(16 x 16)")
    plt.xlabel("Filter Index")
    plt.ylabel("Filter Index")
    
    plt.tight_layout()
    plt.savefig('filter_spectra_correlation.png', dpi=300)