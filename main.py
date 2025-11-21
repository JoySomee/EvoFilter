import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import datetime
from tqdm import tqdm

# 导入用户提供的模块
# 注意：确保 fitness.py, initial.py, population_ops.py 在同一目录下
from initial import FilterInitializer
from fitness import FitnessEvaluator
from population_ops import EvolutionaryOperator

class GeneticAlgorithm:
    def __init__(self, 
                 pop_size=50,  # 种群大小
                 generations=30,  # 最大迭代次数
                 mutation_rate=0.2,  # 变异率
                 crossover_rate=0.8,  # 交叉率
                 replacement_rate=0.1):  # 替换率
        """
        初始化遗传算法参数
        """
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.replacement_rate = replacement_rate
        
        # 检查环境数据
        if not os.path.exists('nk_map.pkl') or not os.path.exists('database_spe.mat'):
            raise FileNotFoundError('关键数据缺失')
        
        self.results_dir = f"results/{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"已创建结果目录: {self.results_dir}")
        
        self.log_file = os.path.join(self.results_dir, "evolution_log.txt")
        with open(self.log_file, 'w') as f:
            # 写入表头，使用竖线分隔和固定宽度对齐
            header = (f"{'Gen':<5} | {'Best Fit':<12} | {'Avg Fit':<12} | "
                      f"{'Uncorr':<10} | {'Info(Raw)':<12} | {'Info(Norm)':<10} | {'Cond':<10}\n")
            f.write(header)
            f.write("-" * 75 + "\n") # 分割线
        print(f"日志将实时记录至: {self.log_file}")
        # 实例化组件
        print("初始化模块...")
        self.initializer = FilterInitializer()
        self.evaluator = FitnessEvaluator() # 会加载 database_spe.mat
        self.operator = EvolutionaryOperator(self.initializer, mutation_sigma=10.0)
        
        # 历史记录
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_uncorr': [],
            'best_info': []
        }
        self.best_individual = None
        self.best_fitness_val = -np.inf
        self.best_results_dict = None
        # 用于动态归一化的边界值
        self.global_logdet_max = -1e9
        self.global_logdet_min = 1e9
    
    def update_normalization_bounds(self, raw_results):
        """更新全局 LogDet 的最大最小值"""
        valid_logdets = []
        for res in raw_results:
            val = res['score_information']
            if val != -np.inf and not np.isnan(val):
                valid_logdets.append(val)
        
        if not valid_logdets:
            return

        current_max = np.max(valid_logdets)
        current_min = np.min(valid_logdets)
        
        if current_max > self.global_logdet_max:
            self.global_logdet_max = current_max
        if current_min < self.global_logdet_min:
            self.global_logdet_min = current_min
        
        if self.global_logdet_max == self.global_logdet_min:
            self.global_logdet_max += 1e-6
    
    def normalize_logdet(self, val):
        """将 LogDet 映射到 [0, 1] 区间"""
        if val == -np.inf or np.isnan(val):
            return 0.0 # 最差得分
        # 归一化公式: (x - min) / (max - min)
        denom = self.global_logdet_max - self.global_logdet_min
        norm_val = (val - self.global_logdet_min) / denom
        # 限制在 [0, 1] 范围内 (因为可能某些旧个体会稍微越界，虽然这里是用全局极值应该不会)
        return np.clip(norm_val, 0.0, 1.0)

    def calculate_scalar_fitness(self, result_dict):
        """
        将多目标指标融合为单目标适应度。
        Fitness = w1 * Uncorrelation + w2 * Normalized_Information
        """
        score_u = result_dict['score_uncorrelation'] # 0 ~ 1
        # 2. 获取并归一化 LogDet (映射到 0~1)
        raw_logdet = result_dict['score_information']
        score_i_norm = self.normalize_logdet(raw_logdet)
        score_s = result_dict['stability_score']
        fitness = 1.0 * score_u + 0.8 * score_i_norm + 0.2 * score_s
        return fitness

    def initialize_population(self):
        print(f"初始化种群 (大小: {self.pop_size})...")
        population = []
        for _ in range(self.pop_size):
            filters, _ = self.initializer.generate_individual_4x4()
            population.append(filters)
        return np.array(population)

    def tournament_selection(self, population, fitnesses, k=3):
        """锦标赛选择"""
        selected_indices = []
        for _ in range(len(population)):
            # 随机选择 k 个个体
            candidates_indices = np.random.choice(len(population), k, replace=False)
            # 找出其中适应度最高的
            best_idx = candidates_indices[np.argmax(fitnesses[candidates_indices])]
            selected_indices.append(best_idx)
        return population[selected_indices]

    def run(self):
        # 1. 初始化种群
        population = self.initialize_population()
        print(f"开始进化，共 {self.generations} 代...")
        for gen in tqdm(range(self.generations)):
            # --- 评估 ---
            fitness_values = []
            raw_results_list = []
            for ind in population:
                res = self.evaluator.evaluate(ind)
                raw_results_list.append(res)
            
            self.update_normalization_bounds(raw_results_list)
            for res in raw_results_list:
                fit = self.calculate_scalar_fitness(res)
                fitness_values.append(fit)
            fitness_values = np.array(fitness_values)
            
            # --- 统计与记录 ---
            current_best_idx = np.argmax(fitness_values)
            current_best_fit = fitness_values[current_best_idx]
            current_avg_fit = np.mean(fitness_values)
            best_res = raw_results_list[current_best_idx]

            # 更新全局最优
            if current_best_fit > self.best_fitness_val:
                self.best_fitness_val = current_best_fit
                self.best_individual = copy.deepcopy(population[current_best_idx])
                self.best_results_dict = best_res
                """保存每一代的最优个体"""
                npy_path = os.path.join(self.results_dir, f"best_individual_{gen}.npy")
                np.save(npy_path, self.best_individual)
                print(f"[Gen {gen}] New Best: {self.best_fitness_val:.4f} "
                      f"(已保存至 {npy_path})")
                print(f"[Gen {gen}] New Best Fitness: {self.best_fitness_val:.4f} "
                      f"(Uncorr: {self.best_results_dict['score_uncorrelation']:.4f}, "
                      f"Info: {self.best_results_dict['score_information']:.2f})")
            
            self.history['best_fitness'].append(current_best_fit)
            self.history['avg_fitness'].append(current_avg_fit)
            self.history['best_uncorr'].append(best_res['score_uncorrelation'])
            self.history['best_info'].append(best_res['score_information'])
            
            with open(self.log_file, 'a') as f:
                # 使用 f-string 进行对齐控制: <12 表示左对齐占12格, .4f 表示保留4位小数
                log_line = (f"{gen:<5} | "
                            f"{current_best_fit:<12.4f} | "
                            f"{current_avg_fit:<12.4f} | "
                            f"{best_res['score_uncorrelation']:<10.4f} | "
                            f"{best_res['score_information']:<12.2f} | "
                            f"{self.normalize_logdet(best_res['score_information']):<10.4f} | "
                            f"{best_res['stability_score']:<10.4f}\n")
                f.write(log_line)
            
            # 打印进度
            if gen % 5 == 0:
                print(f"Generation {gen}: Avg Fit = {current_avg_fit:.2f}, Best Fit = {current_best_fit:.2f}")

            # --- 精英保留 (Elitism) ---
            # 保留最好的 2 个个体直接进入下一代
            sorted_indices = np.argsort(fitness_values)[::-1] # 降序
            elites = population[sorted_indices[:2]]
            
            # --- 选择 ---
            # 选出用于产生后代的父代池
            parents = self.tournament_selection(population, fitness_values)
            
            # --- 交叉与变异 ---
            next_population = []
            # 先放入精英
            for elite in elites:
                next_population.append(elite)
            
            # 填充剩余位置 (pop_size - len(elites))
            while len(next_population) < self.pop_size:
                # 随机选两个父代
                p1_idx = np.random.randint(len(parents))
                p2_idx = np.random.randint(len(parents))
                parent1 = parents[p1_idx]
                parent2 = parents[p2_idx]
                # 交叉
                if np.random.rand() < self.crossover_rate:
                    child = self.operator.crossover(parent1, parent2)
                else:
                    child = parent1.copy() # 没交叉就直接复制
                # 变异
                child = self.operator.mutation(child, probability=self.mutation_rate)
                # 随机替换 (引入新血统)
                child = self.operator.random_replacement(child, probability=self.replacement_rate)
                next_population.append(child)
            population = np.array(next_population)

        print("\n进化完成!")
        self.plot_history()
        self.plot_best_result()

    def plot_history(self):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Uncorrelation Score (0-1)', color=color)
        ax1.plot(self.history['best_uncorr'], color=color, label='Uncorrelation', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(linestyle=':', alpha=0.5)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Information (LogDet)', color=color)
        ax2.plot(self.history['best_info'], color=color, label='LogDet (Raw)', linestyle='--', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Evolutionary Trajectory of Objectives')
        fig.tight_layout() 
        
        # 保存到 results 目录
        save_path = os.path.join(self.results_dir, 'evolution_history.png')
        plt.savefig(save_path)
        print(f"历史曲线已保存至 {save_path}")

    def plot_best_result(self):
        if self.best_results_dict is None:
            return
        wavelengths = np.linspace(400, 700, 301)
        res = self.best_results_dict
        spectra = res['spectra']

        # 1. 将16条光谱曲线绘制到4x4的子图中
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle(f"Best Solution Spectra (Uncorrelation Score: {res['score_uncorrelation']:.3f})", fontsize=16)

        for i, ax in enumerate(axes.flat):
            if i < len(spectra):
                ax.plot(wavelengths, spectra[i])
                ax.set_title(f"Spectrum {i+1}")
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("Transmittance")
                ax.grid(True, linestyle=':')
            else:
                ax.axis('off')  # 隐藏多余的子图

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        spectra_save_path = os.path.join(self.results_dir, 'best_solution_spectra.png')
        plt.savefig(spectra_save_path)
        print(f"最优光谱曲线图已保存至 {spectra_save_path}")
        plt.close(fig)

        # 2. 将相关系数矩阵单独保存为一张图
        plt.figure(figsize=(8, 6))
        plt.imshow(res['correlation_matrix'], cmap='coolwarm', vmin=0, vmax=1, origin='lower')
        plt.colorbar(label="Correlation")
        plt.title("Correlation Matrix")
        
        corr_matrix_save_path = os.path.join(self.results_dir, 'correlation_matrix.png')
        plt.savefig(corr_matrix_save_path)
        plt.close()
        print(f"相关系数矩阵图已保存至 {corr_matrix_save_path}")

if __name__ == "__main__":
    # 运行算法
    ga = GeneticAlgorithm(
        pop_size=100,        # 种群大小
        generations=50,     # 迭代代数
        mutation_rate=0.2,  # 变异率
        crossover_rate=0.8  # 交叉率
    )
    ga.run()
