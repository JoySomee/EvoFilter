import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import datetime
import argparse
import yaml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

# 导入用户提供的模块
from initial import FilterInitializer
from fitness import FitnessEvaluator
from population_ops import EvolutionaryOperator
from fitness_functions import (
    get_fitness_function,
    list_fitness_functions,
    print_fitness_functions,
    FitnessConfig
)

# 全局变量用于多进程
_global_evaluator = None
_global_corr_mode = 'row'

def _init_worker(corr_mode='row'):
    """初始化工作进程的全局评估器"""
    global _global_evaluator, _global_corr_mode
    _global_corr_mode = corr_mode
    _global_evaluator = FitnessEvaluator(corr_mode=corr_mode)

def _evaluate_individual(individual):
    """工作进程中评估单个个体"""
    global _global_evaluator
    return _global_evaluator.evaluate(individual)


def load_config(config_path='config.yaml'):
    """
    从 YAML 配置文件加载参数
    """
    if not os.path.exists(config_path):
        print(f"警告: 配置文件 {config_path} 不存在，使用默认参数")
        return None

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class GeneticAlgorithm:
    def __init__(self,
                 pop_size=50,  # 种群大小
                 generations=30,  # 最大迭代次数
                 mutation_rate=0.2,  # 变异率
                 crossover_rate=0.8,  # 交叉率
                 replacement_rate=0.1,  # 替换率
                 num_workers=None,  # 并行进程数，默认为CPU核数
                 fitness_type='combined',  # 适应度函数类型
                 corr_mode='row',  # 不相关性计算模式
                 # === 适应度函数权重 (用于 combined 类型) ===
                 w_info=1.0,
                 w_uncorr=0.0,
                 w_stability=0.0,
                 # === 惩罚项参数 ===
                 trans_threshold=0.5,      # 透过率阈值
                 lambda_trans=10.0,        # 透过率惩罚系数
                 corr_threshold=0.7,       # 互相关阈值 (最大相关系数)
                 lambda_corr=20.0,         # 互相关惩罚系数
                 lambda_uniformity=5.0,    # 均匀性惩罚系数
                 cond_threshold=100.0,     # 条件数阈值 (死区边界)
                 lambda_stability=5.0):    # 稳定性惩罚系数
        """
        初始化遗传算法参数

        适应度函数类型 (--fitness_type):
        - 'information': 信息量优先，最大化 D-Optimality
        - 'uncorrelation': 去相关性优先，最小化滤光片间相关性
        - 'stability': 稳定性优先，最小化条件数
        - 'combined': 组合适应度 (默认)，加权组合多目标
        - 'weighted_sum': 简单加权和，归一化指标加权
        - 'pareto': Pareto 前沿适应度

        不相关性计算模式 (--corr_mode):
        - 'row': 行不相关性，计算滤光片之间的相关性 (16x16矩阵)
        - 'column': 列不相关性，计算波长之间的相关性 (WxW矩阵)
        - 'both': 同时计算行和列，取较差的分数

        使用 'python src/main.py --list_fitness' 查看所有可用的适应度函数
        """
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.replacement_rate = replacement_rate
        self.num_workers = num_workers if num_workers else max(1, cpu_count() - 1)

        # 适应度函数类型和权重
        self.fitness_type = fitness_type
        self.corr_mode = corr_mode
        self.w_info = w_info
        self.w_uncorr = w_uncorr
        self.w_stability = w_stability

        # 惩罚项参数
        self.trans_threshold = trans_threshold
        self.lambda_trans = lambda_trans
        self.corr_threshold = corr_threshold
        self.lambda_corr = lambda_corr
        self.lambda_uniformity = lambda_uniformity
        self.cond_threshold = cond_threshold
        self.lambda_stability = lambda_stability

        # 初始化适应度函数
        fitness_config = FitnessConfig(
            trans_threshold=trans_threshold,
            lambda_trans=lambda_trans,
            corr_threshold=corr_threshold,
            lambda_corr=lambda_corr,
            lambda_uniformity=lambda_uniformity,
            cond_threshold=cond_threshold,
            lambda_stability=lambda_stability
        )
        self.fitness_function = get_fitness_function(
            fitness_type,
            config=fitness_config,
            w_info=w_info,
            w_uncorr=w_uncorr,
            w_stability=w_stability
        )
        print(f"使用适应度函数: {self.fitness_function.name}")
        print(f"不相关性计算模式: {corr_mode}")

        # 检查环境数据
        if not os.path.exists('data/nk_map.pkl') or not os.path.exists('data/database_spe.mat'):
            raise FileNotFoundError('关键数据缺失')

        self.results_dir = f"results/{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"已创建结果目录: {self.results_dir}")

        self.log_file = os.path.join(self.results_dir, "evolution_log.txt")
        with open(self.log_file, 'w') as f:
            # 写入表头
            header = (f"{'Gen':<5} | {'Best Fit':<10} | {'Avg Fit':<10} | "
                      f"{'Info':<8} | {'MaxCorr':<8} | {'MeanTrans':<10} | "
                      f"{'StdTrans':<10} | {'CondNum':<12} | "
                      f"{'P_trans':<8} | {'P_corr':<8} | {'P_unif':<8} | {'P_stab':<8}\n")
            f.write(header)
            f.write("-" * 135 + "\n")
        print(f"日志将实时记录至: {self.log_file}")

        # 实例化组件
        print("初始化模块...")
        self.initializer = FilterInitializer()
        self.evaluator = FitnessEvaluator(corr_mode=corr_mode)
        self.operator = EvolutionaryOperator(self.initializer, mutation_sigma=15.0)

        # 历史记录
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_info': [],
            'best_max_corr': [],
            'best_mean_trans': [],
            'best_cond_number': []
        }
        self.best_individual = None
        self.best_fitness_val = -np.inf
        self.best_results_dict = None
    
    def calculate_scalar_fitness(self, result_dict):
        """
        计算标量适应度值 (使用模块化适应度函数)

        返回:
            fitness: 标量适应度值
            penalties: 包含各分项的字典 (用于日志记录)
        """
        fitness, details = self.fitness_function(result_dict)

        # 为了兼容日志格式，确保返回标准化的 penalties 字典
        penalties = {
            'p_trans': details.get('p_trans', 0),
            'p_corr': details.get('p_corr', 0),
            'p_unif': details.get('p_unif', 0),
            'p_stab': details.get('p_stab', 0)
        }

        return fitness, penalties

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
        print(f"开始进化，共 {self.generations} 代... (使用 {self.num_workers} 个进程并行评估)")

        # 创建进程池，传递 corr_mode 参数给 worker
        pool = Pool(processes=self.num_workers,
                    initializer=_init_worker,
                    initargs=(self.corr_mode,))

        try:
            for gen in tqdm(range(self.generations)):
                # --- 并行评估 ---
                fitness_values = []
                penalties_list = []

                # 使用进程池并行计算所有个体的适应度
                raw_results_list = pool.map(_evaluate_individual, list(population))

                for res in raw_results_list:
                    fit, penalties = self.calculate_scalar_fitness(res)
                    fitness_values.append(fit)
                    penalties_list.append(penalties)
                fitness_values = np.array(fitness_values)

                # --- 统计与记录 ---
                current_best_idx = np.argmax(fitness_values)
                current_best_fit = fitness_values[current_best_idx]
                current_avg_fit = np.mean(fitness_values)
                best_res = raw_results_list[current_best_idx]
                best_penalties = penalties_list[current_best_idx]

                # 提取关键指标
                best_info = best_res['score_information']
                best_max_corr = 1.0 - best_res['score_uncorrelation']
                best_trans_details = best_res['transmittance_details']
                best_mean_trans = best_trans_details['overall_mean_transmittance']
                best_std_trans = best_trans_details['channel_std_transmittance']
                best_cond_number = best_res['condition_number']

                # 更新全局最优
                if current_best_fit > self.best_fitness_val:
                    self.best_fitness_val = current_best_fit
                    self.best_individual = copy.deepcopy(population[current_best_idx])
                    self.best_results_dict = best_res
                    # 保存最优个体
                    npy_path = os.path.join(self.results_dir, f"best_individual_{gen}.npy")
                    np.save(npy_path, self.best_individual)
                    print(f"\n[Gen {gen}] New Best Fitness: {self.best_fitness_val:.4f}")
                    print(f"    Info: {best_info:.2f}, MaxCorr: {best_max_corr:.4f}, "
                          f"MeanTrans: {best_mean_trans:.4f}, CondNum: {best_cond_number:.2f}")
                    print(f"    Penalties - Trans: {best_penalties['p_trans']:.4f}, "
                          f"Corr: {best_penalties['p_corr']:.4f}, "
                          f"Unif: {best_penalties['p_unif']:.4f}, "
                          f"Stab: {best_penalties['p_stab']:.4f}")

                # 更新历史记录
                self.history['best_fitness'].append(current_best_fit)
                self.history['avg_fitness'].append(current_avg_fit)
                self.history['best_info'].append(best_info)
                self.history['best_max_corr'].append(best_max_corr)
                self.history['best_mean_trans'].append(best_mean_trans)
                self.history['best_cond_number'].append(best_cond_number)

                # 写入日志
                with open(self.log_file, 'a') as f:
                    log_line = (f"{gen:<5} | "
                                f"{current_best_fit:<10.4f} | "
                                f"{current_avg_fit:<10.4f} | "
                                f"{best_info:<8.2f} | "
                                f"{best_max_corr:<8.4f} | "
                                f"{best_mean_trans:<10.4f} | "
                                f"{best_std_trans:<10.4f} | "
                                f"{best_cond_number:<12.2f} | "
                                f"{best_penalties['p_trans']:<8.4f} | "
                                f"{best_penalties['p_corr']:<8.4f} | "
                                f"{best_penalties['p_unif']:<8.4f} | "
                                f"{best_penalties['p_stab']:<8.4f}\n")
                    f.write(log_line)

                # 打印进度
                if gen % 5 == 0:
                    print(f"Generation {gen}: Avg Fit = {current_avg_fit:.4f}, Best Fit = {current_best_fit:.4f}")

                # --- 每10代保存一次整个种群 ---
                if gen % 10 == 0:
                    population_save_path = os.path.join(self.results_dir, f'population_gen_{gen}.npy')
                    np.save(population_save_path, population)

                # --- 精英保留 (Elitism) ---
                sorted_indices = np.argsort(fitness_values)[::-1]
                elites = population[sorted_indices[:2]]

                # --- 选择 ---
                parents = self.tournament_selection(population, fitness_values)

                # --- 交叉与变异 ---
                next_population = []
                for elite in elites:
                    next_population.append(elite)

                while len(next_population) < self.pop_size:
                    p1_idx = np.random.randint(len(parents))
                    p2_idx = np.random.randint(len(parents))
                    parent1 = parents[p1_idx]
                    parent2 = parents[p2_idx]
                    # 交叉
                    if np.random.rand() < self.crossover_rate:
                        child = self.operator.crossover(parent1, parent2)
                    else:
                        child = parent1.copy()
                    # 变异
                    progress = float(gen) / self.generations
                    child = self.operator.mutation_sparse_adaptive(child, probability=self.mutation_rate, progress=progress)
                    # 随机替换
                    child = self.operator.random_replacement(child, probability=self.replacement_rate)
                    next_population.append(child)
                population = np.array(next_population)

        finally:
            # 关闭进程池
            pool.close()
            pool.join()

        print("\n进化完成!")
        final_population_path = os.path.join(self.results_dir, 'final_population.npy')
        np.save(final_population_path, population)
        print(f"最后一轮所有子代已保存至 {final_population_path}")

        self.plot_history()
        self.plot_best_result()

    def plot_history(self):
        """绘制进化历史曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 图1: 适应度曲线
        ax1 = axes[0, 0]
        ax1.plot(self.history['best_fitness'], 'b-', label='Best Fitness', linewidth=2)
        ax1.plot(self.history['avg_fitness'], 'b--', label='Avg Fitness', alpha=0.7)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution')
        ax1.legend()
        ax1.grid(linestyle=':', alpha=0.5)

        # 图2: 信息量曲线
        ax2 = axes[0, 1]
        ax2.plot(self.history['best_info'], 'g-', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Information Score (LogDet)')
        ax2.set_title('Information Score Evolution')
        ax2.grid(linestyle=':', alpha=0.5)

        # 图3: 最大相关系数 & 透过率
        ax3 = axes[1, 0]
        ax3.plot(self.history['best_max_corr'], 'r-', label='Max Correlation', linewidth=2)
        ax3.plot(self.history['best_mean_trans'], 'm-', label='Mean Transmittance', linewidth=2)
        ax3.axhline(y=self.corr_threshold, color='r', linestyle='--', alpha=0.5, label=f'Corr Threshold ({self.corr_threshold})')
        ax3.axhline(y=self.trans_threshold, color='m', linestyle='--', alpha=0.5, label=f'Trans Threshold ({self.trans_threshold})')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Score')
        ax3.set_title('Correlation & Transmittance')
        ax3.legend()
        ax3.grid(linestyle=':', alpha=0.5)

        # 图4: 条件数 (对数尺度)
        ax4 = axes[1, 1]
        ax4.semilogy(self.history['best_cond_number'], 'c-', linewidth=2)
        ax4.axhline(y=self.cond_threshold, color='r', linestyle='--', alpha=0.7,
                   label=f'Threshold ({self.cond_threshold})')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Condition Number (log scale)')
        ax4.set_title('Numerical Stability (Condition Number)')
        ax4.legend()
        ax4.grid(linestyle=':', alpha=0.5)

        plt.suptitle('Evolutionary Trajectory', fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'evolution_history.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"历史曲线已保存至 {save_path}")

    def plot_best_result(self):
        """绘制最优解的详细结果"""
        if self.best_results_dict is None:
            return
        wavelengths = np.linspace(400, 700, 301)
        res = self.best_results_dict
        spectra = res['spectra']
        trans_details = res['transmittance_details']

        # 1. 16条光谱曲线
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        max_corr = 1.0 - res['score_uncorrelation']
        fig.suptitle(f"Best Solution Spectra\n"
                     f"Info: {res['score_information']:.2f}, MaxCorr: {max_corr:.4f}, "
                     f"MeanTrans: {trans_details['overall_mean_transmittance']:.4f}", fontsize=14)

        channel_means = trans_details['channel_mean_transmittance']
        for i, ax in enumerate(axes.flat):
            if i < len(spectra):
                ax.plot(wavelengths, spectra[i], 'b-', linewidth=1.5)
                ax.axhline(y=channel_means[i], color='r', linestyle='--', alpha=0.7,
                          label=f'Mean: {channel_means[i]:.3f}')
                ax.set_title(f"Filter {i+1}")
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("Transmittance")
                ax.set_ylim([0, 1])
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, linestyle=':')
            else:
                ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        spectra_save_path = os.path.join(self.results_dir, 'best_solution_spectra.png')
        plt.savefig(spectra_save_path, dpi=150)
        plt.close(fig)
        print(f"最优光谱曲线图已保存至 {spectra_save_path}")

        # 2. 相关系数矩阵
        plt.figure(figsize=(10, 8))
        im = plt.imshow(res['correlation_matrix'], cmap='coolwarm', vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, label="Correlation Coefficient")
        plt.title(f"Filter Correlation Matrix (Max: {max_corr:.4f})")
        plt.xlabel("Filter Index")
        plt.ylabel("Filter Index")

        corr_matrix_save_path = os.path.join(self.results_dir, 'correlation_matrix.png')
        plt.savefig(corr_matrix_save_path, dpi=150)
        plt.close()
        print(f"相关系数矩阵图已保存至 {corr_matrix_save_path}")

        # 3. 通道透过率柱状图
        plt.figure(figsize=(10, 5))
        x = np.arange(16)
        plt.bar(x, channel_means, color='steelblue', alpha=0.8)
        plt.axhline(y=trans_details['overall_mean_transmittance'], color='r', linestyle='--',
                   label=f"Overall Mean: {trans_details['overall_mean_transmittance']:.4f}")
        plt.axhline(y=self.trans_threshold, color='orange', linestyle='--',
                   label=f"Threshold: {self.trans_threshold}")
        plt.xlabel("Filter Channel")
        plt.ylabel("Mean Transmittance")
        plt.title(f"Channel Transmittance Distribution (Std: {trans_details['channel_std_transmittance']:.4f})")
        plt.xticks(x, [f"F{i+1}" for i in range(16)])
        plt.legend()
        plt.grid(axis='y', linestyle=':', alpha=0.5)

        trans_save_path = os.path.join(self.results_dir, 'channel_transmittance.png')
        plt.savefig(trans_save_path, dpi=150)
        plt.close()
        print(f"通道透过率分布图已保存至 {trans_save_path}")


def get_params(config_path='config.yaml'):
    """
    从配置文件加载参数，命令行参数可选覆盖

    优先级: 命令行参数 > 配置文件 > 默认值
    """
    # 默认参数
    defaults = {
        'pop_size': 100,
        'generations': 100,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'replacement_rate': 0.1,
        'num_workers': None,  # None表示自动检测CPU核数
        # 适应度函数配置
        'fitness_type': 'combined',
        'corr_mode': 'row',  # 不相关性计算模式: row, column, both
        'w_info': 1.0,
        'w_uncorr': 0.0,
        'w_stability': 0.0,
        # 惩罚项参数
        'trans_threshold': 0.5,
        'lambda_trans': 10.0,
        'corr_threshold': 0.7,
        'lambda_corr': 20.0,
        'lambda_uniformity': 5.0,
        'cond_threshold': 100.0,
        'lambda_stability': 5.0
    }

    # 从配置文件加载
    config = load_config(config_path)
    if config:
        # GA 基本参数
        if 'ga' in config:
            for key in ['pop_size', 'generations', 'mutation_rate', 'crossover_rate', 'replacement_rate', 'num_workers']:
                if key in config['ga']:
                    defaults[key] = config['ga'][key]
        # 适应度函数配置
        if 'fitness' in config:
            for key in ['fitness_type', 'corr_mode', 'w_info', 'w_uncorr', 'w_stability']:
                if key in config['fitness']:
                    defaults[key] = config['fitness'][key]
        # 惩罚项参数
        if 'penalties' in config:
            for key in ['trans_threshold', 'lambda_trans', 'corr_threshold', 'lambda_corr',
                        'lambda_uniformity', 'cond_threshold', 'lambda_stability']:
                if key in config['penalties']:
                    defaults[key] = config['penalties'][key]

    # 命令行参数解析 (可选覆盖)
    available_fitness = list_fitness_functions()
    parser = argparse.ArgumentParser(
        description='遗传算法优化滤光片设计\n'
                    '参数优先从 config.yaml 读取，命令行参数可覆盖配置文件\n\n'
                    f'可用适应度函数: {", ".join(available_fitness)}',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径 (default: config.yaml)')
    parser.add_argument('--list_fitness', action='store_true',
                        help='列出所有可用的适应度函数并退出')

    # 适应度函数参数
    fitness_group = parser.add_argument_group('适应度函数配置')
    fitness_group.add_argument('--fitness_type', type=str, default=None,
                        choices=available_fitness,
                        help=f'适应度函数类型 (可选: {", ".join(available_fitness)})')
    fitness_group.add_argument('--corr_mode', type=str, default=None,
                        choices=['row', 'column', 'both'],
                        help='不相关性计算模式: row=滤光片间, column=波长间, both=两者取较差')
    fitness_group.add_argument('--w_info', type=float, default=None,
                        help='信息量权重 (用于 combined 类型)')
    fitness_group.add_argument('--w_uncorr', type=float, default=None,
                        help='去相关性权重 (用于 combined 类型)')
    fitness_group.add_argument('--w_stability', type=float, default=None,
                        help='稳定性权重 (用于 combined 类型)')

    # GA基本参数
    ga_group = parser.add_argument_group('遗传算法参数')
    ga_group.add_argument('--pop_size', type=int, default=None, help='种群大小')
    ga_group.add_argument('--generations', type=int, default=None, help='迭代代数')
    ga_group.add_argument('--mutation_rate', type=float, default=None, help='变异率')
    ga_group.add_argument('--crossover_rate', type=float, default=None, help='交叉率')
    ga_group.add_argument('--replacement_rate', type=float, default=None, help='替换率')
    ga_group.add_argument('--num_workers', type=int, default=None, help='并行进程数 (默认: CPU核数-1)')

    # 惩罚项参数
    penalty_group = parser.add_argument_group('惩罚项参数')
    penalty_group.add_argument('--trans_threshold', type=float, default=None, help='透过率阈值')
    penalty_group.add_argument('--lambda_trans', type=float, default=None, help='透过率惩罚系数')
    penalty_group.add_argument('--corr_threshold', type=float, default=None, help='最大相关系数阈值')
    penalty_group.add_argument('--lambda_corr', type=float, default=None, help='互相关惩罚系数')
    penalty_group.add_argument('--lambda_uniformity', type=float, default=None, help='通道均匀性惩罚系数')
    penalty_group.add_argument('--cond_threshold', type=float, default=None, help='条件数阈值')
    penalty_group.add_argument('--lambda_stability', type=float, default=None, help='稳定性惩罚系数')

    args = parser.parse_args()

    # 如果用户请求列出适应度函数，打印后退出
    if args.list_fitness:
        print_fitness_functions()
        exit(0)

    # 命令行参数覆盖配置文件
    for key in defaults.keys():
        cmd_val = getattr(args, key, None)
        if cmd_val is not None:
            defaults[key] = cmd_val

    return defaults


if __name__ == "__main__":
    params = get_params()

    # 打印当前参数
    print("=" * 60)
    print("当前运行参数:")
    print("-" * 60)
    print("[遗传算法]")
    print(f"  种群大小: {params['pop_size']}")
    print(f"  迭代代数: {params['generations']}")
    print(f"  变异率: {params['mutation_rate']}")
    print(f"  交叉率: {params['crossover_rate']}")
    print(f"  替换率: {params['replacement_rate']}")
    print(f"  并行进程数: {params['num_workers'] if params['num_workers'] else '自动 (CPU核数-1)'}")
    print("-" * 60)
    print("[适应度函数]")
    print(f"  类型: {params['fitness_type']}")
    print(f"  不相关性模式: {params['corr_mode']}")
    if params['fitness_type'] == 'combined':
        print(f"  权重: w_info={params['w_info']}, w_uncorr={params['w_uncorr']}, w_stability={params['w_stability']}")
    print("-" * 60)
    print("[惩罚项参数]")
    print(f"  透过率阈值: {params['trans_threshold']}, λ={params['lambda_trans']}")
    print(f"  相关系数阈值: {params['corr_threshold']}, λ={params['lambda_corr']}")
    print(f"  均匀性惩罚: λ={params['lambda_uniformity']}")
    print(f"  条件数阈值: {params['cond_threshold']}, λ={params['lambda_stability']}")
    print("=" * 60)

    ga = GeneticAlgorithm(
        pop_size=params['pop_size'],
        generations=params['generations'],
        mutation_rate=params['mutation_rate'],
        crossover_rate=params['crossover_rate'],
        replacement_rate=params['replacement_rate'],
        num_workers=params['num_workers'],
        fitness_type=params['fitness_type'],
        corr_mode=params['corr_mode'],
        w_info=params['w_info'],
        w_uncorr=params['w_uncorr'],
        w_stability=params['w_stability'],
        trans_threshold=params['trans_threshold'],
        lambda_trans=params['lambda_trans'],
        corr_threshold=params['corr_threshold'],
        lambda_corr=params['lambda_corr'],
        lambda_uniformity=params['lambda_uniformity'],
        cond_threshold=params['cond_threshold'],
        lambda_stability=params['lambda_stability']
    )
    ga.run()
