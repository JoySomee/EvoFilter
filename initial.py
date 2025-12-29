import numpy as np
import matplotlib.pyplot as plt
import pickle

class FilterInitializer:
    def __init__(self, num_layers=20, min_total=400.0, max_total=2000.0):
        """
        初始化滤光片生成器配置
        :param num_layers: 层数 (默认20)
        :param min_total: 最小总厚度 (nm)
        :param max_total: 最大总厚度 (nm)
        """
        self.num_layers = num_layers
        self.min_total = min_total
        self.max_total = max_total
        self.min_layer_thickness = 20   # 单层最小 20nm
        self.max_layer_thickness = 200  # 单层最大 200nm
        
        # 近似折射率 (用于物理启发式初始化，实际TMM计算时会用准确的色散数据)
        # 假设偶数层为 SiO2, 奇数层为 TiO2
        with open('nk_map.pkl', 'rb') as f:
            self.nk_list_map = pickle.load(f)
        
    # def validate_and_fix(self, thicknesses):
    #     """
    #     约束强制函数：
    #     1. 缩放比例以适应范围
    #     2. 四舍五入为整数
    #     3. 确保单层 >= 1nm
    #     4. 微调总厚度以满足 [min, max]
    #     """
    #     # --- 第一步：浮点数级别的预缩放 (保持物理结构比例) ---
    #     # 防止负数
    #     # 强行控制每一层的厚度在 [min_layer_thickness, max_layer_thickness] 之间
    #     t = np.round(thicknesses).astype(int)
    #     t = np.clip(t, self.min_layer_thickness, self.max_layer_thickness)
    #     thicknesses = np.maximum(thicknesses, 0.0)
    #     current_total_float = np.sum(thicknesses)
        
    #     # 如果总厚度为0（极端情况），随机重置
    #     if current_total_float < 1e-6:
    #         thicknesses = np.ones(self.num_layers) * (self.min_total / self.num_layers)
    #         current_total_float = self.min_total

    #     # 缩放到范围内 (稍微留一点余量给取整误差)
    #     if current_total_float < self.min_total:
    #         thicknesses *= (self.min_total / current_total_float) * 1.01
    #     elif current_total_float > self.max_total:
    #         thicknesses *= (self.max_total / current_total_float) * 0.99

    #     # --- 第二步：离散化 ---
    #     thicknesses_int = np.round(thicknesses).astype(int)
        
    #     # --- 第三步：硬约束修正 ---
    #     # 1. 确保每层至少 1nm
    #     thicknesses_int = np.maximum(thicknesses_int, 1)
        
    #     # 2. 检查总和并进行整数级微调
    #     current_total = np.sum(thicknesses_int)
        
    #     if current_total < self.min_total:
    #         # 还需要增加 diff 纳米
    #         diff = int(self.min_total - current_total)
    #         # 随机选择 diff 个位置，每个加 1nm
    #         indices = np.random.choice(self.num_layers, diff, replace=True)
    #         for idx in indices:
    #             thicknesses_int[idx] += 1
                
    #     elif current_total > self.max_total:
    #         # 还需要减少 diff 纳米
    #         diff = int(current_total - self.max_total)
    #         # 随机选择层减 1nm，但必须保证减完后 >= 1nm
    #         while diff > 0:
    #             idx = np.random.randint(0, self.num_layers)
    #             if thicknesses_int[idx] > 1:
    #                 thicknesses_int[idx] -= 1
    #                 diff -= 1
            
    #     return thicknesses_int
    
    def validate_and_fix(self, thicknesses):
        """
        核心修正函数：严格保证每一层在 [20, 200] 之间，且总和 <= 2000
        """
        # 1. 预处理：取整
        t = np.round(thicknesses).astype(int)
        
        # 2. 第一轮钳位 (Hard Clamp): 强行把每一层限制在 [20, 200]
        t = np.clip(t, self.min_layer_thickness, self.max_layer_thickness)
        
        # 3. 检查总厚度
        current_total = np.sum(t)
        
        # 情况 A: 总厚度超过 2000nm
        if current_total > self.max_total:
            # 我们需要减少的总量
            reduce_amount = current_total - self.max_total
            
            # 策略：我们不能简单地减去所有层，因为不能低于 20nm。
            # 我们只能从那些 "富余" (thickness > 20) 的部分里扣除。
            
            # 计算每层有多少"油水"可以刮 (即超过 20nm 的部分)
            excess = t - self.min_layer_thickness
            total_excess = np.sum(excess)
            
            if total_excess == 0:
                # 理论上不会发生，除非 max_total 设置得比 min_possible 还小
                return t 

            # 计算压缩比例：我们需要保留多少比例的"油水"
            # 目标总余量 = 允许的最大总余量 (2000 - 20*N)
            target_excess_sum = self.max_total - (self.min_layer_thickness * self.num_layers)
            scale = target_excess_sum / total_excess
            
            # 重新计算厚度：基础值 + 缩放后的余量
            new_excess = excess * scale
            t = self.min_layer_thickness + new_excess
            
            # 再次取整并钳位 (防止浮点误差导致越界)
            t = np.round(t).astype(int)
            t = np.clip(t, self.min_layer_thickness, self.max_layer_thickness)
            
            # 4. 最后的整数微调 (因为上面取整可能导致总和略微变化)
            # 如果依然 > 2000，随机选择大于20的层减1
            while np.sum(t) > self.max_total:
                # 找出所有还能减的层 ( > 20)
                indices = np.where(t > self.min_layer_thickness)[0]
                idx = np.random.choice(indices)
                t[idx] -= 1
        
        # 情况 B: 总厚度太小？
        # 由于我们第一步做了 clip(t, 20, 200)，且 N=20
        # 最小总厚度自然是 400nm。只要 400 < 2000，就不需要处理下限。
        
        return t

    def generate_random(self):
        """
        策略1: 约束随机初始化
        完全随机分配厚度，只满足总和约束。
        """
        # 1. 随机选择一个目标总厚度
        target_total = np.random.uniform(self.min_total, self.max_total)
        # 2. 生成 N 个随机权重 (Dirichlet分布的思想)
        weights = np.random.rand(self.num_layers)
        # 3. 归一化并分配
        thicknesses = (weights / np.sum(weights)) * target_total
        return self.validate_and_fix(thicknesses)

    def generate_qwot(self, noise_level=0.2):
        """
        策略2: 四分之一波长堆栈 (QWOT) + 噪声
        物理含义: d = lambda / (4 * n)
        """
        # 1. 随机选择一个中心波长 (覆盖可见光到近红外)
        center_wavelength = np.random.randint(400, 701)
        # 2. 计算理论物理厚度 d = lambda / 4n
        base_thicknesses = center_wavelength / (4 * np.array(self.nk_list_map[center_wavelength][:-1]))
        # 3. 添加高斯噪声 (Jitter)，避免过于完美的晶体结构
        # noise_level = 0.2 表示 20% 的标准差抖动
        noise = np.random.normal(1, noise_level, self.num_layers)
        thicknesses = base_thicknesses * noise
        return self.validate_and_fix(thicknesses)

    def generate_fabry_perot(self, noise_level=0.1):
        """
        策略3: 法布里-珀罗腔 (Fabry-Perot)
        结构: [反射镜堆栈] - [厚间隔层] - [反射镜堆栈]
        """
        # 1. 基础 QWOT 结构
        center_wavelength = np.random.randint(500, 601)
        thicknesses = center_wavelength / (4 * np.array(self.nk_list_map[center_wavelength][:-1]))
        
        # 2. 选择中间的一层或两层作为 "腔" (Spacer)
        # 将中间层的厚度变为原来的 4倍 到 8倍 (即 1波长 或 2波长厚度)
        mid_idx = self.num_layers // 2
        spacer_factor = np.random.uniform(4.0, 8.0)
        # 修改中间两层 (一层SiO2一层TiO2) 以保持交替结构的连续性
        thicknesses[mid_idx] *= spacer_factor
        thicknesses[mid_idx-1] *= spacer_factor
        # 3. 添加较小的噪声
        noise = np.random.normal(1, noise_level, self.num_layers)
        thicknesses *= noise
        return self.validate_and_fix(thicknesses)

    def generate_individual_4x4(self):
        """
        生成一个个体，包含 16 个滤光片。
        采用混合策略：
        - 约 8 个 Random (提供多样性)
        - 约 4 个 QWOT (提供强反射/截止带)
        - 约 4 个 FP (提供窄带透射)
        """
        filters = []
        labels = [] # 用于记录每个位置是什么类型的，方便调试/可视化
        
        # 定义策略池
        # 我们创建一个包含 16 个策略标签的列表，然后打乱它
        strategies = ['random'] * 16 + ['qwot'] * 0 + ['fp'] * 0
        np.random.shuffle(strategies)
        
        for strategy in strategies:
            if strategy == 'random':
                filters.append(self.generate_random())
                labels.append("Rnd")
            elif strategy == 'qwot':
                filters.append(self.generate_qwot())
                labels.append("QWOT")
            elif strategy == 'fp':
                filters.append(self.generate_fabry_perot())
                labels.append("FP")
        # 将列表转换为 (16, 20) 的 numpy 数组
        # 形状: [滤光片数量, 层数]
        return np.array(filters), np.array(labels).reshape(4, 4)

# --- 运行示例与可视化 ---
if __name__ == "__main__":
    # 设置
    initializer = FilterInitializer(num_layers=20, min_total=400, max_total=2000)
    print("正在初始化一个 4x4 滤光片个体...")
    individual_filters, type_labels = initializer.generate_individual_4x4()
    print(f"个体形状: {individual_filters.shape}") # 应该是 (16, 20)
    print("个体生成完毕。开始绘制...")
    # 设置绘图
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    layers = np.arange(1, 21)
    # 遍历 4x4 网格进行绘图
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            current_filter = individual_filters[idx]
            filter_type = type_labels[i, j]
            ax = axes[i, j]
            # 根据类型设置不同的颜色主题，方便肉眼区分
            if filter_type == "Rnd":
                color = 'gray'
                title_color = 'black'
            elif filter_type == "QWOT":
                # 蓝色/粉色交替，代表 SiO2/TiO2
                color = ['skyblue' if k%2==0 else 'salmon' for k in range(20)]
                title_color = 'blue'
            else: # FP
                color = 'purple' # 紫色突出显示腔体
                title_color = 'purple'

            ax.barh(layers, current_filter, color=color, alpha=0.8)
            
            # 标注总厚度
            total_t = np.sum(current_filter)
            ax.set_title(f"({i},{j}) {filter_type}\nTotal: {total_t}nm", 
                         fontsize=10, color=title_color, fontweight='bold')
            
            # 只在边缘显示轴标签
            if i == 3:
                ax.set_xlabel("Thickness (nm)")
            if j == 0:
                ax.set_ylabel("Layer Index")
                
            ax.grid(axis='x', linestyle=':', alpha=0.5)

    plt.suptitle("Initialized Individual: 4x4 Filter Array Structure", fontsize=16)
    plt.savefig("4x4_filter_array.png")
    
    print("可视化完成。这就代表了进化算法中的'一个个体'。")
    
    
