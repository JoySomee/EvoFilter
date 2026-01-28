import numpy as np
from initial import FilterInitializer


class EvolutionaryOperator:
    def __init__(self, initializer, mutation_sigma=15.0):
        """
        :param initializer: 之前定义的 FilterInitializer 实例 (用于约束校验和生成新基因)
        :param mutation_sigma: 高斯变异的标准差 (nm)
        """
        self.initializer = initializer
        self.initial_sigma = mutation_sigma

    def crossover(self, parent_a, parent_b):
        """
        【操作一：交叉】滤光片级均匀交叉 (Filter-level Uniform Crossover)
        
        原理：
        像抛硬币一样，对于子代的每一个位置 (共16个)，
        50%概率继承父亲的滤光片，50%概率继承母亲的滤光片。
        
        :param parent_a: 父代个体 (16, N)
        :param parent_b: 母代个体 (16, N)
        :return: 子代个体 (16, N)
        """
        num_filters = parent_a.shape[0]
        # 生成一个掩码 (True/False)，形状为 (16,)
        # mask 为 True 的位置继承 A，False 继承 B
        mask = np.random.rand(num_filters) < 0.5
        # 初始化子代
        offspring = np.empty_like(parent_a)
        # 应用掩码复制基因
        offspring[mask] = parent_a[mask]
        offspring[~mask] = parent_b[~mask]
        return offspring

    def mutation(self, individual, probability=0.1):
        """
        【操作二：变异】层厚度级高斯变异 (Layer-level Gaussian Mutation)
        原理：
        对个体中的每一个滤光片，以一定概率进行微调。
        微调方式是加上一个整数高斯噪声，然后强制约束修正。
        
        :param individual: 待变异个体 (16, N)
        :param probability: 每个滤光片发生变异的概率 (0.0 - 1.0)
        :return: 变异后的新个体 (copy)
        """
        # 复制一份，避免修改原数据
        offspring = individual.copy()
        num_filters, num_layers = offspring.shape
        
        for i in range(num_filters):
            # 判定该滤光片是否发生变异
            if np.random.rand() < probability:
                
                # --- 变异逻辑 ---
                # 1. 生成变异向量 (所有层都加一点噪声，或者只随机选几层)
                # 这里策略是：每一层都有可能微小变动，模拟制造误差或微调
                # round() 保证变异量是整数
                delta = np.round(np.random.normal(0, self.mutation_sigma, num_layers)).astype(int)
                
                # 2. 应用变异
                new_thicknesses = offspring[i] + delta
                
                # 3. 关键：约束校验与修正 (Validate and Fix)
                # 调用 Initializer 中的方法来处理 <0 和 总厚度越界问题
                # 并确保返回的是整数
                offspring[i] = self.initializer.validate_and_fix(new_thicknesses)
                
        return offspring
    
    def mutation_sparse_adaptive(self, individual, probability=0.1, progress=0.0):
        """
        【操作二：稀疏自适应变异】 (Sparse Adaptive Mutation)
        改进点：
        1. 自适应步长：随着 progress (0->1) 增加，变异幅度 sigma 减小。
        2. 稀疏变异：不再修改所有层，而是随机选择少数几层进行修改。
        
        :param individual: 待变异个体
        :param probability: 变异概率
        :param progress: 当前进化进度 (Current_Gen / Max_Gen)，范围 0.0 到 1.0
        :return: 变异后的个体
        """
        offspring = individual.copy()
        num_filters, num_layers = offspring.shape
        
        # 1. 计算动态步长 (非均匀变异)
        # 公式：sigma = base * (1 - progress)^2
        # 例如：开始是 15nm，进度 50% 时是 3.75nm，结束时接近 1nm
        current_sigma = self.initial_sigma * ((1.0 - progress) ** 2)
        # 保证最小有 1.0nm 的变动能力 (防止后期变为0)
        current_sigma = max(1.0, current_sigma)
        
        for i in range(num_filters):
            if np.random.rand() < probability:
                
                # --- 改进逻辑：稀疏选择 ---
                # 随机决定要修改多少层 (1 到 5 层之间，而不是全部 20 层)
                # 这种"微创手术"比"全身整容"更容易产生优良后代
                num_changes = np.random.randint(1, 6) 
                
                # 随机选择要修改的层索引
                layer_indices = np.random.choice(num_layers, num_changes, replace=False)
                
                # 生成噪声
                delta = np.random.normal(0, current_sigma, num_changes)
                delta = np.round(delta).astype(int)
                
                # 应用变异
                new_thicknesses = offspring[i].copy()
                new_thicknesses[layer_indices] += delta
                
                # 约束校验
                offspring[i] = self.initializer.validate_and_fix(new_thicknesses)
                
        return offspring
    

    def random_replacement(self, individual, probability=0.05):
        """
        【操作三：随机替换】单点宏观变异 (Filter Replacement)
        
        原理：
        为了防止局部最优，随机“踢掉”阵列中的某一个滤光片，
        用一个全新的、随机生成的滤光片（Random/QWOT/FP）取而代之。
        这相当于引入了“外来移民”。
        
        :param individual: 待操作个体 (16, N)
        :param probability: 发生替换操作的概率
        :return: 操作后的新个体
        """
        offspring = individual.copy()
        num_filters = offspring.shape[0]
        
        # 判定整个个体是否执行替换操作
        if np.random.rand() < probability:
            
            # 1. 随机选择阵列中的一个位置 (0-15)
            idx_to_replace = np.random.randint(0, num_filters)
            # 2. 随机选择一种生成策略 (Random, QWOT, 或 FP)
            # 这里我们简单地随机调用 initializer 的方法
            strategy = np.random.choice(['random', 'qwot', 'fp'])
            if strategy == 'random':
                new_filter = self.initializer.generate_random()
            elif strategy == 'qwot':
                new_filter = self.initializer.generate_qwot()
            else:
                new_filter = self.initializer.generate_fabry_perot()
            # 3. 替换
            offspring[idx_to_replace] = new_filter
        return offspring

# ==========================================
# 单元测试与使用示例
# ==========================================
if __name__ == "__main__":
    # 初始化
    initializer = FilterInitializer()
    operator = EvolutionaryOperator(initializer, mutation_sigma=15.0)
    
    # 创建两个父代 (全是 10nm 和 全是 100nm)
    parent1 = np.ones((16, 20), dtype=int) * 10
    parent2 = np.ones((16, 20), dtype=int) * 100
    
    print("--- 测试交叉 ---")
    child_cross = operator.crossover(parent1, parent2)
    print(f"Parent1[0,0]: {parent1[0,0]}, Parent2[0,0]: {parent2[0,0]}")
    print(f"Child 第一列 (应包含10和100):\n{child_cross[:, 0]}")
    
    print("\n--- 测试变异 ---")
    # 强制变异概率 1.0 来观察效果
    child_mut = operator.mutation(parent1, probability=1.0)
    print(f"原值: {parent1[0, :5]}")
    print(f"变异: {child_mut[0, :5]}")
    print("差异:", child_mut[0, :5] - parent1[0, :5])
    
    print("\n--- 测试替换 ---")
    # 强制替换
    child_rep = operator.random_replacement(parent1, probability=1.0)
    # 找出哪一行被替换了
    diff_rows = np.where(np.any(child_rep != parent1, axis=1))[0]
    print(f"被替换的滤光片索引: {diff_rows}")
    if len(diff_rows) > 0:
        print(f"新滤光片前5层: {child_rep[diff_rows[0], :5]}")