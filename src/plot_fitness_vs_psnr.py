"""
绘制最终种群中每个子代的适应度值 vs 重建PSNR的散点图
支持三种模式：
1. 从种群文件计算并绘图
2. 从已保存的npz文件直接绘图
3. 从多代种群文件中随机抽取个体绘图
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse
import os
import yaml
import glob
from multiprocessing import Pool, cpu_count
from cal_response import cal_response_batch
from evaluate import generate_filters_and_measurement, reconstruction_pipeline, numpy_psnr

# ============ 多进程支持 ============
# 全局变量用于工作进程
_worker_evaluator = None
_worker_test_data = None
_worker_penalty_params = None


def _init_worker(test_data, penalty_params):
    """初始化工作进程的全局变量"""
    global _worker_evaluator, _worker_test_data, _worker_penalty_params
    from fitness import FitnessEvaluator
    _worker_evaluator = FitnessEvaluator()
    _worker_test_data = test_data
    _worker_penalty_params = penalty_params


def _evaluate_individual_worker(args):
    """
    工作进程中评估单个个体
    返回: (fitness, psnr, penalties)
    """
    global _worker_evaluator, _worker_test_data, _worker_penalty_params

    individual, idx, total = args

    # 计算适应度
    result = _worker_evaluator.evaluate(individual)
    fitness, penalties = calculate_scalar_fitness(result, **_worker_penalty_params)

    # 计算重建 PSNR
    psnr = evaluate_psnr_for_individual(individual, _worker_test_data, _worker_evaluator)

    return {
        'idx': idx,
        'fitness': fitness,
        'psnr': psnr,
        'penalties': penalties
    }


def _evaluate_individual_with_gen_worker(args):
    """
    工作进程中评估单个个体（带代数信息）
    返回: dict with fitness, psnr, penalties, gen_num
    """
    global _worker_evaluator, _worker_test_data, _worker_penalty_params

    individual, idx, gen_num = args

    # 计算适应度
    result = _worker_evaluator.evaluate(individual)
    fitness, penalties = calculate_scalar_fitness(result, **_worker_penalty_params)

    # 计算重建 PSNR
    psnr = evaluate_psnr_for_individual(individual, _worker_test_data, _worker_evaluator)

    return {
        'idx': idx,
        'gen_num': gen_num,
        'fitness': fitness,
        'psnr': psnr,
        'penalties': penalties
    }


def load_config(config_path='config.yaml'):
    """从 YAML 配置文件加载参数"""
    if not os.path.exists(config_path):
        return None
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def calculate_scalar_fitness(result_dict,
                              trans_threshold=0.5, lambda_trans=10.0,
                              corr_threshold=0.7, lambda_corr=20.0,
                              lambda_uniformity=5.0,
                              cond_threshold=100.0, lambda_stability=5.0):
    """
    计算标量适应度值（与 main.py 保持一致）

    适应度 = 信息量收益 - 透过率惩罚 - 互相关惩罚 - 均匀性惩罚 - 稳定性惩罚
    """
    # === 正向收益: 信息量 (D-Optimality) ===
    raw_info = result_dict['score_information']
    if raw_info == -np.inf or np.isnan(raw_info):
        return -1e6, {'p_trans': 0, 'p_corr': 0, 'p_unif': 0, 'p_stab': 0, 'info': 0}

    reward_info = raw_info

    # === 惩罚1: 透过率门控 (均值平方惩罚) ===
    trans_details = result_dict['transmittance_details']
    mean_trans = trans_details['overall_mean_transmittance']
    if mean_trans < trans_threshold:
        penalty_trans = lambda_trans * ((trans_threshold - mean_trans) ** 2)
    else:
        penalty_trans = 0.0

    # === 惩罚2: 互相关性 (阈值门控，超过阈值重罚) ===
    max_corr = 1.0 - result_dict['score_uncorrelation']
    if max_corr > corr_threshold:
        penalty_corr = lambda_corr * ((max_corr - corr_threshold) ** 2)
    else:
        penalty_corr = 0.0

    # === 惩罚3: 通道均匀性 (标准差平方惩罚) ===
    channel_std = trans_details['channel_std_transmittance']
    penalty_uniformity = lambda_uniformity * (channel_std ** 2)

    # === 惩罚4: 稳定性 (带死区的合页损失) ===
    cond_number = result_dict['condition_number']
    log_cond = np.log10(cond_number) if cond_number > 0 else 0
    log_threshold = np.log10(cond_threshold)

    if log_cond <= log_threshold:
        penalty_stability = 0.0
    else:
        penalty_stability = lambda_stability * ((log_cond - log_threshold) ** 2)

    # === 综合适应度 ===
    fitness = reward_info - penalty_trans - penalty_corr - penalty_uniformity - penalty_stability

    penalties = {
        'p_trans': penalty_trans,
        'p_corr': penalty_corr,
        'p_unif': penalty_uniformity,
        'p_stab': penalty_stability,
        'info': reward_info,
        'max_corr': max_corr,
        'mean_trans': mean_trans,
        'cond_number': cond_number
    }

    return fitness, penalties


def calculate_mean_transmittance(individual):
    """
    计算单个个体的平均透过率
    """
    spectra = cal_response_batch(individual)  # (16, 301)
    # 计算所有滤光片在所有波长上的平均透过率
    mean_trans = np.mean(spectra)
    return mean_trans


def evaluate_psnr_for_individual(individual, test_data, evaluator):
    """
    对单个个体进行重建并计算平均 PSNR
    """
    
    # 获取光谱响应矩阵
    bands = [453.3, 457.6, 462.1, 466.8, 471.6, 476.5, 481.6, 
         486.9, 492.4, 498.0, 503.9, 509.9, 516.2, 522.7, 
         529.5, 536.5, 543.8, 551.4, 558.6, 567.5, 575.3, 
         584.3, 594.4, 604.2, 614.4, 625.1, 636.3, 648.1]
    spectra = cal_response_batch(individual, bands, 'data/nk_map_special.pkl')  # (16, 28)
    
    # 重塑为 4x4 的 mask
    mask_3d = spectra.reshape(4, 4, 28)
    
    n_image, H, W, n_bands = test_data.shape
    all_psnr = []
    
    for i in range(n_image):
        # 生成测量值
        raw_image = generate_filters_and_measurement(test_data[i], mask_3d)
        # 重建
        x_recon_flat = reconstruction_pipeline(raw_image, mask_3d.reshape(-1, n_bands))
        x_recon = x_recon_flat.reshape(H, W, n_bands)
        
        # 计算 PSNR
        pred_chw = x_recon.transpose(2, 0, 1)
        truth_chw = test_data[i].transpose(2, 0, 1)
        psnr_val = numpy_psnr(pred_chw, truth_chw)
        all_psnr.append(psnr_val)
    
    return np.mean(all_psnr)


def plot_from_npz(args):
    """从npz文件读取数据并绘图"""
    print(f">>> 从npz文件加载数据: {args.npz}")
    data = np.load(args.npz)
    
    # 打印可用的字段
    print(f"    可用字段: {list(data.keys())}")
    
    psnr_values = np.array(data['psnr']).flatten()
    transmittance_values = np.array(data['transmittance']).flatten()
    
    # 根据参数选择横轴数据
    x_axis_options = {
        'fitness': ('fitness', 'Fitness Value'),
        'uncorr': ('score_u', 'Uncorrelation Score'),
        'info': ('score_i_norm', 'Information Score (Normalized)'),
        'stability': ('score_s', 'Stability Score'),
        'transmittance': ('transmittance', 'Mean Transmittance')
    }
    
    if args.x_axis not in x_axis_options:
        print(f"错误: 无效的横轴选项 '{args.x_axis}'")
        print(f"可用选项: {list(x_axis_options.keys())}")
        return
    
    field_name, x_label = x_axis_options[args.x_axis]
    
    if field_name not in data:
        print(f"错误: npz文件中不存在字段 '{field_name}'")
        return
    
    x_values = np.array(data[field_name]).flatten()
    
    # 检查数据维度是否匹配
    if x_values.shape[0] != psnr_values.shape[0]:
        print(f"错误: 横轴数据维度 ({x_values.shape[0]}) 与 PSNR 数据维度 ({psnr_values.shape[0]}) 不匹配")
        print(f"提示: 该 npz 文件中 '{field_name}' 可能只保存了标量值，请使用 --x_axis fitness 或重新生成数据")
        return
    
    print(f"    x_values shape: {x_values.shape}, psnr_values shape: {psnr_values.shape}")
    
    # 计算相关系数
    correlation = np.corrcoef(x_values, psnr_values)[0, 1]
    print(f"\n>>> {x_label} 与 PSNR 的相关系数: {correlation:.4f}")
    
    # 绘制散点图（按平均透过率上色）
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_values, psnr_values, c=transmittance_values, 
                          cmap='viridis', alpha=0.7, edgecolors='black', s=80)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Mean Transmittance', fontsize=12)
    
    # 添加趋势线
    z = np.polyfit(x_values, psnr_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_values.min(), x_values.max(), 100)
    plt.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (r={correlation:.3f})')
    
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Reconstruction PSNR (dB)', fontsize=14)
    plt.title(f'{x_label} vs Reconstruction PSNR', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"\n>>> 图片已保存至: {args.output}")


def compute_and_plot_from_generations(args, penalty_params):
    """从多代种群文件中每代独立抽取个体计算数据并绘图"""
    from fitness import FitnessEvaluator
    from evaluate import load_real_data
    from cal_response import cal_response_batch

    # 查找所有种群文件
    print(f">>> 在目录中查找种群文件: {args.results_dir}")
    population_pattern = os.path.join(args.results_dir, "population_gen_*.npy")
    population_files = glob.glob(population_pattern)

    # 查找最终种群文件
    final_population_file = os.path.join(args.results_dir, "final_population.npy")
    has_final_population = os.path.exists(final_population_file) and args.include_final

    if not population_files and not has_final_population:
        print(f"错误: 在 {args.results_dir} 中未找到种群文件")
        return

    # 按代数排序
    population_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    print(f"    找到 {len(population_files)} 个中间代种群文件:")
    for f in population_files:
        gen_num = os.path.basename(f).split('_')[-1].split('.')[0]
        print(f"      第 {gen_num} 代: {f}")

    if has_final_population:
        print(f"    找到最终种群文件: {final_population_file}")

    # 设置随机种子以保证可重现性
    np.random.seed(args.seed)

    # 从每代独立抽取个体
    sampled_individuals = []
    sampled_generations = []

    # 处理中间代种群文件
    for pop_file in population_files:
        gen_num = int(os.path.basename(pop_file).split('_')[-1].split('.')[0])
        population = np.load(pop_file)
        pop_size = len(population)

        # 从当前代抽取个体
        n_sample_per_gen = min(args.n_sample_per_gen, pop_size)
        sample_indices = np.random.choice(pop_size, n_sample_per_gen, replace=False)

        print(f"    第 {gen_num} 代: 从 {pop_size} 个个体中抽取 {n_sample_per_gen} 个")

        for idx in sample_indices:
            sampled_individuals.append(population[idx])
            sampled_generations.append(gen_num)

    # 处理最终种群文件
    if has_final_population:
        final_population = np.load(final_population_file)
        final_pop_size = len(final_population)

        # 从最终种群抽取个体
        n_sample_final = min(args.n_sample_per_gen, final_pop_size)
        final_sample_indices = np.random.choice(final_pop_size, n_sample_final, replace=False)

        print(f"    最终代: 从 {final_pop_size} 个个体中抽取 {n_sample_final} 个")

        for idx in final_sample_indices:
            sampled_individuals.append(final_population[idx])
            sampled_generations.append(-1)  # 用-1标记最终代

    total_samples = len(sampled_individuals)
    n_generations = len(population_files)
    if has_final_population:
        print(f"    总共抽取 {total_samples} 个个体进行评估 ({n_generations} 个中间代 + 1 个最终代，每代 {args.n_sample_per_gen} 个)")
    else:
        print(f"    总共抽取 {total_samples} 个个体进行评估 ({n_generations} 代 × {args.n_sample_per_gen} 个/代)")

    # 加载测试数据
    print(f">>> 加载测试数据: {args.test_data}")
    test_data = load_real_data(args.test_data)
    print(f"    测试图像数量: {test_data.shape[0]}")

    # 确定并行进程数
    num_workers = args.num_workers if args.num_workers else max(1, cpu_count() - 1)
    print(f">>> 使用 {num_workers} 个进程并行评估 {total_samples} 个个体...")

    # 准备任务参数 (individual, idx, gen_num)
    tasks = [(individual, idx, gen_num)
             for idx, (individual, gen_num) in enumerate(zip(sampled_individuals, sampled_generations))]

    # 使用进程池并行计算
    from functools import partial
    init_func = partial(_init_worker, test_data, penalty_params)

    # 初始化结果列表
    fitness_values = [None] * total_samples
    psnr_values = [None] * total_samples
    transmittance_values = [None] * total_samples
    info_list = [None] * total_samples
    max_corr_list = [None] * total_samples
    cond_number_list = [None] * total_samples
    generation_list = [None] * total_samples

    with Pool(processes=num_workers, initializer=init_func) as pool:
        # 使用 imap_unordered 获取实时进度
        from tqdm import tqdm
        results = list(tqdm(
            pool.imap_unordered(_evaluate_individual_with_gen_worker, tasks),
            total=total_samples,
            desc="评估进度"
        ))

    # 整理结果（按原始顺序）
    for res in results:
        idx = res['idx']
        fitness_values[idx] = res['fitness']
        psnr_values[idx] = res['psnr']
        penalties = res['penalties']
        generation_list[idx] = res['gen_num']
        transmittance_values[idx] = penalties['mean_trans']
        info_list[idx] = penalties['info']
        max_corr_list[idx] = penalties['max_corr']
        cond_number_list[idx] = penalties['cond_number']

    fitness_values = np.array(fitness_values)
    psnr_values = np.array(psnr_values)
    transmittance_values = np.array(transmittance_values)
    info_arr = np.array(info_list)
    max_corr_arr = np.array(max_corr_list)
    cond_number_arr = np.array(cond_number_list)
    generation_arr = np.array(generation_list)

    # 计算相关系数
    correlation = np.corrcoef(fitness_values, psnr_values)[0, 1]
    print(f"\n>>> 适应度与 PSNR 的相关系数: {correlation:.4f}")

    # 绘制散点图（按代数上色，最终代特殊处理）
    plt.figure(figsize=(12, 8))

    # 分离中间代和最终代
    intermediate_mask = generation_arr != -1
    final_mask = generation_arr == -1

    # 绘制中间代个体
    if np.any(intermediate_mask):
        intermediate_gens = generation_arr[intermediate_mask]
        intermediate_fitness = fitness_values[intermediate_mask]
        intermediate_psnr = psnr_values[intermediate_mask]

        unique_gens = np.unique(intermediate_gens)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_gens)))

        for i, gen in enumerate(unique_gens):
            mask = intermediate_gens == gen
            plt.scatter(intermediate_fitness[mask], intermediate_psnr[mask],
                       c=[colors[i]], label=f'Gen {gen}',
                       alpha=0.7, edgecolors='black', s=80)

    # 绘制最终代个体（用特殊标记）
    if np.any(final_mask):
        final_fitness = fitness_values[final_mask]
        final_psnr = psnr_values[final_mask]
        plt.scatter(final_fitness, final_psnr,
                   c='red', marker='*', s=120,
                   label='Final Generation',
                   alpha=0.8, edgecolors='darkred', linewidth=1)

    # 添加趋势线
    z = np.polyfit(fitness_values, psnr_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(fitness_values.min(), fitness_values.max(), 100)
    plt.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (r={correlation:.3f})')

    plt.xlabel('Fitness Value', fontsize=14)
    plt.ylabel('Reconstruction PSNR (dB)', fontsize=14)
    plt.title(f'Fitness vs Reconstruction PSNR ({args.n_sample_per_gen} samples per generation)', fontsize=16)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n>>> 图片已保存至: {args.output}")

    # 保存数据
    data_path = args.output.replace('.png', '_data.npz')
    np.savez(data_path,
             fitness=fitness_values,
             psnr=psnr_values,
             transmittance=transmittance_values,
             generation=generation_arr,
             correlation=correlation,
             info=info_arr,
             max_corr=max_corr_arr,
             cond_number=cond_number_arr)
    print(f">>> 数据已保存至: {data_path}")


def compute_and_plot(args, penalty_params):
    """从种群文件计算数据并绘图（并行版本）"""
    from fitness import FitnessEvaluator
    from evaluate import load_real_data
    from cal_response import cal_response_batch
    from functools import partial

    # 加载最终种群
    print(f">>> 加载种群文件: {args.population}")
    population = np.load(args.population)
    pop_size = len(population)
    print(f"    种群大小: {pop_size}")

    # 加载测试数据
    print(f">>> 加载测试数据: {args.test_data}")
    test_data = load_real_data(args.test_data)
    print(f"    测试图像数量: {test_data.shape[0]}")

    # 确定并行进程数
    num_workers = args.num_workers if args.num_workers else max(1, cpu_count() - 1)
    print(f">>> 使用 {num_workers} 个进程并行评估 {pop_size} 个个体...")

    # 准备任务参数
    tasks = [(individual, idx, pop_size) for idx, individual in enumerate(population)]

    # 使用进程池并行计算
    # 注意: 使用 initializer 来初始化每个工作进程的全局变量
    init_func = partial(_init_worker, test_data, penalty_params)

    fitness_values = [None] * pop_size
    psnr_values = [None] * pop_size
    transmittance_values = [None] * pop_size
    info_list = [None] * pop_size
    max_corr_list = [None] * pop_size
    cond_number_list = [None] * pop_size

    with Pool(processes=num_workers, initializer=init_func) as pool:
        # 使用 imap_unordered 获取实时进度
        from tqdm import tqdm
        results = list(tqdm(
            pool.imap_unordered(_evaluate_individual_worker, tasks),
            total=pop_size,
            desc="评估进度"
        ))

    # 整理结果（按原始顺序）
    for res in results:
        idx = res['idx']
        fitness_values[idx] = res['fitness']
        psnr_values[idx] = res['psnr']
        penalties = res['penalties']
        transmittance_values[idx] = penalties['mean_trans']
        info_list[idx] = penalties['info']
        max_corr_list[idx] = penalties['max_corr']
        cond_number_list[idx] = penalties['cond_number']

    fitness_values = np.array(fitness_values)
    psnr_values = np.array(psnr_values)
    transmittance_values = np.array(transmittance_values)
    info_arr = np.array(info_list)
    max_corr_arr = np.array(max_corr_list)
    cond_number_arr = np.array(cond_number_list)

    # 计算相关系数
    correlation = np.corrcoef(fitness_values, psnr_values)[0, 1]
    print(f"\n>>> 适应度与 PSNR 的相关系数: {correlation:.4f}")

    # 绘制散点图（按平均透过率上色）
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(fitness_values, psnr_values, c=transmittance_values,
                          cmap='viridis', alpha=0.7, edgecolors='black', s=80)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Mean Transmittance', fontsize=12)

    # 添加趋势线
    z = np.polyfit(fitness_values, psnr_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(fitness_values.min(), fitness_values.max(), 100)
    plt.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (r={correlation:.3f})')

    plt.xlabel('Fitness Value', fontsize=14)
    plt.ylabel('Reconstruction PSNR (dB)', fontsize=14)
    plt.title('Fitness vs Reconstruction PSNR for Final Population', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"\n>>> 图片已保存至: {args.output}")

    # 保存数据
    data_path = args.output.replace('.png', '_data.npz')
    np.savez(data_path,
             fitness=fitness_values,
             psnr=psnr_values,
             transmittance=transmittance_values,
             correlation=correlation,
             info=info_arr,
             max_corr=max_corr_arr,
             cond_number=cond_number_arr)
    print(f">>> 数据已保存至: {data_path}")


def main():
    parser = argparse.ArgumentParser(description='绘制适应度 vs PSNR 散点图')

    # 模式选择：从npz读取 或 从种群计算 或 从多代种群抽样
    parser.add_argument('--npz', type=str, default=None,
                        help='从npz文件读取数据绘图（优先级最高）')
    parser.add_argument('--x_axis', type=str, default='fitness',
                        choices=['fitness', 'uncorr', 'info', 'stability', 'transmittance'],
                        help='横轴数据选择: fitness/uncorr/info/stability/transmittance')

    # 从多代种群抽样的参数
    parser.add_argument('--results_dir', '-r', type=str, default=None,
                        help='结果目录路径，包含多个 population_gen_*.npy 文件')
    parser.add_argument('--n_sample_per_gen', '-n', type=int, default=20,
                        help='从每一代中抽取的个体数量 (default: 20)')
    parser.add_argument('--include_final', action='store_true',
                        help='是否包含最终种群 (final_population.npy)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，用于保证结果可重现 (default: 42)')

    # 从单个种群计算的参数
    parser.add_argument('--population', '-p', type=str, default=None,
                        help='单个种群文件路径 (final_population.npy)')
    parser.add_argument('--test_data', '-t', type=str,
                        default='../dataset/TSA_simu_data/Truth/',
                        help='测试数据路径')

    # 并行计算参数
    parser.add_argument('--num_workers', '-w', type=int, default=None,
                        help='并行进程数 (默认: CPU核数-1)')

    # 惩罚项参数（与 main.py 和 config.yaml 保持一致）
    parser.add_argument('--trans_threshold', type=float, default=0.5,
                        help='透过率阈值 (default: 0.5)')
    parser.add_argument('--lambda_trans', type=float, default=3000.0,
                        help='透过率惩罚系数 (default: 3000.0)')
    parser.add_argument('--corr_threshold', type=float, default=0.3,
                        help='最大相关系数阈值 (default: 0.3)')
    parser.add_argument('--lambda_corr', type=float, default=20.0,
                        help='互相关惩罚系数 (default: 20.0)')
    parser.add_argument('--lambda_uniformity', type=float, default=20.0,
                        help='均匀性惩罚系数 (default: 20.0)')
    parser.add_argument('--cond_threshold', type=float, default=100.0,
                        help='条件数阈值 (default: 100.0)')
    parser.add_argument('--lambda_stability', type=float, default=2.0,
                        help='稳定性惩罚系数 (default: 2.0)')

    # 配置文件
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='配置文件路径 (default: config.yaml)')

    # 输出参数
    parser.add_argument('--output', '-o', type=str, default='fitness_vs_psnr.png',
                        help='输出图片路径')

    args = parser.parse_args()

    # 尝试从配置文件加载参数
    config = load_config(args.config)

    # 加载 num_workers（从配置文件或命令行）
    if args.num_workers is None and config and 'ga' in config:
        args.num_workers = config['ga'].get('num_workers', None)

    if config and 'penalties' in config:
        penalties_config = config['penalties']
        # 如果命令行没有指定，则使用配置文件的值
        penalty_params = {
            'trans_threshold': penalties_config.get('trans_threshold', args.trans_threshold),
            'lambda_trans': penalties_config.get('lambda_trans', args.lambda_trans),
            'corr_threshold': penalties_config.get('corr_threshold', args.corr_threshold),
            'lambda_corr': penalties_config.get('lambda_corr', args.lambda_corr),
            'lambda_uniformity': penalties_config.get('lambda_uniformity', args.lambda_uniformity),
            'cond_threshold': penalties_config.get('cond_threshold', args.cond_threshold),
            'lambda_stability': penalties_config.get('lambda_stability', args.lambda_stability),
        }
        print(f">>> 从配置文件加载惩罚参数: {args.config}")
    else:
        # 使用命令行参数
        penalty_params = {
            'trans_threshold': args.trans_threshold,
            'lambda_trans': args.lambda_trans,
            'corr_threshold': args.corr_threshold,
            'lambda_corr': args.lambda_corr,
            'lambda_uniformity': args.lambda_uniformity,
            'cond_threshold': args.cond_threshold,
            'lambda_stability': args.lambda_stability,
        }
        print(">>> 使用命令行惩罚参数")

    print(f"    惩罚参数: {penalty_params}")
    print(f"    并行进程数: {args.num_workers if args.num_workers else '自动 (CPU核数-1)'}")

    if args.npz:
        # 模式1: 从npz文件读取并绘图
        plot_from_npz(args)
    elif args.results_dir:
        # 模式2: 从多代种群文件中随机抽取个体绘图
        compute_and_plot_from_generations(args, penalty_params)
    elif args.population:
        # 模式3: 从单个种群文件计算并绘图
        compute_and_plot(args, penalty_params)
    else:
        print("错误: 请指定以下参数之一:")
        print("  --npz: 从npz文件读取数据")
        print("  --results_dir: 从多代种群文件中抽样")
        print("  --population: 从单个种群文件计算")
        parser.print_help()


if __name__ == "__main__":
    main()
