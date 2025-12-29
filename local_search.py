import numpy as np
import copy

class LocalSearchOptimizer:
    def __init__(self, initializer, evaluator, fitness_calculator):
        """
        局部搜索优化器 (基于随机爬山法)
        
        :param initializer: FilterInitializer 实例，用于校验约束
        :param evaluator: FitnessEvaluator 实例，用于计算物理响应
        :param fitness_calculator: 函数，接收 evaluate 的结果字典，返回单目标标量分数
        """
        self.initializer = initializer
        self.evaluator = evaluator
        self.fitness_calculator = fitness_calculator
        
    def optimize(self, individual, current_fitness, max_steps=20, step_std=2.0):
        """
        执行局部搜索
        
        :param individual: 初始个体 (16, N)
        :param current_fitness: 当前个体的适应度标量
        :param max_steps: 最大尝试步数 (Budget)
        :param step_std: 微调的标准差 (nm)，通常设得很小 (如 1.0 - 3.0)
        :return: (optimized_individual, optimized_fitness, optimized_result_dict)
        """
        # 深拷贝避免修改原个体
        best_ind = copy.deepcopy(individual)
        best_fit = current_fitness
        best_res_dict = None # 缓存最好结果的详细字典，避免外部再次计算
        
        # 获取个体的形状
        rows, cols = best_ind.shape
        
        # 记录是否有所提升
        improved = False
        
        for i in range(max_steps):
            # --- 1. 生成邻域解 (微小扰动) ---
            candidate = best_ind.copy()
            
            # 策略：不是改变所有位置，而是随机挑几个点进行微调
            # 每次只改变约 5% - 10% 的基因，或者至少改变一个
            num_mutations = max(1, int(rows * cols * 0.05)) 
            
            # 随机选择要修改的坐标
            row_indices = np.random.randint(0, rows, num_mutations)
            col_indices = np.random.randint(0, cols, num_mutations)
            
            # 生成微小的整数噪声 (例如 -2, -1, 1, 2)
            noise = np.random.normal(0, step_std, num_mutations)
            noise = np.round(noise).astype(int)
            
            # 加上噪声
            candidate[row_indices, col_indices] += noise
            
            # --- 2. 约束修正 ---
            # 这一步很关键，必须保证微调后的结果依然符合物理约束 (总厚度、单层厚度)
            # 我们逐行进行校验
            for r in np.unique(row_indices):
                candidate[r] = self.initializer.validate_and_fix(candidate[r])
            
            # --- 3. 评估 ---
            # 注意：这里会消耗计算资源，所以 max_steps 不宜过大
            try:
                res_dict = self.evaluator.evaluate(candidate)
                candidate_fit = self.fitness_calculator(res_dict)
                
                # --- 4. 贪婪选择 ---
                if candidate_fit > best_fit:
                    best_ind = candidate
                    best_fit = candidate_fit
                    best_res_dict = res_dict
                    improved = True
                    # 可以在这里加入自适应步长：如果成功了，也许可以稍微大胆一点？
                    # 或者保持微小步长以进行精细挖掘
            except Exception as e:
                print(f"Local search evaluation failed: {e}")
                continue
                
        # 如果没有提升，best_res_dict 可能是 None，需要处理
        if best_res_dict is None and not improved:
            # 如果没提升，通常外部不需要 best_res_dict，或者可以重新计算一次
            # 为了节省计算，这里我们只返回 None，由外部决定是否需要重算
            pass
            
        return best_ind, best_fit, best_res_dict, improved