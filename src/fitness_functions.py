# =============================================================================
# fitness_functions.py - 模块化适应度函数框架
# =============================================================================
# 设计原则:
#   1. 基类定义统一接口，子类实现具体逻辑
#   2. 使用注册表模式，方便动态添加新的适应度函数
#   3. 每个适应度函数可独立配置参数
#
# 添加新适应度函数的步骤:
#   1. 继承 BaseFitnessFunction 类
#   2. 实现 calculate() 方法
#   3. 使用 @register_fitness 装饰器注册
# =============================================================================

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field


# =============================================================================
# 适应度函数注册表
# =============================================================================

_FITNESS_REGISTRY: Dict[str, type] = {}


def register_fitness(name: str):
    """
    装饰器: 注册适应度函数到全局注册表

    使用方法:
        @register_fitness("my_fitness")
        class MyFitness(BaseFitnessFunction):
            ...
    """
    def decorator(cls):
        _FITNESS_REGISTRY[name] = cls
        cls._registered_name = name
        return cls
    return decorator


def get_fitness_function(name: str, **kwargs) -> 'BaseFitnessFunction':
    """
    工厂函数: 根据名称获取适应度函数实例

    Args:
        name: 适应度函数名称
        **kwargs: 传递给适应度函数的参数

    Returns:
        适应度函数实例

    Raises:
        ValueError: 如果名称未注册
    """
    if name not in _FITNESS_REGISTRY:
        available = list(_FITNESS_REGISTRY.keys())
        raise ValueError(f"未知的适应度函数: '{name}'. 可用选项: {available}")
    return _FITNESS_REGISTRY[name](**kwargs)


def list_fitness_functions() -> List[str]:
    """返回所有已注册的适应度函数名称"""
    return list(_FITNESS_REGISTRY.keys())


# =============================================================================
# 适应度函数配置数据类
# =============================================================================

@dataclass
class FitnessConfig:
    """适应度函数通用配置"""
    # 透过率约束
    trans_threshold: float = 0.5
    lambda_trans: float = 10.0

    # 相关性约束
    corr_threshold: float = 0.7
    lambda_corr: float = 20.0

    # 均匀性约束
    lambda_uniformity: float = 5.0

    # 稳定性约束
    cond_threshold: float = 100.0
    lambda_stability: float = 5.0

    # 额外参数 (用于特定适应度函数)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'trans_threshold': self.trans_threshold,
            'lambda_trans': self.lambda_trans,
            'corr_threshold': self.corr_threshold,
            'lambda_corr': self.lambda_corr,
            'lambda_uniformity': self.lambda_uniformity,
            'cond_threshold': self.cond_threshold,
            'lambda_stability': self.lambda_stability,
            **self.extra_params
        }


# =============================================================================
# 适应度函数基类
# =============================================================================

class BaseFitnessFunction(ABC):
    """
    适应度函数基类

    所有适应度函数必须继承此类并实现 calculate() 方法
    """

    _registered_name: str = "base"

    def __init__(self, config: Optional[FitnessConfig] = None, **kwargs):
        """
        初始化适应度函数

        Args:
            config: FitnessConfig 配置对象
            **kwargs: 直接传递的配置参数 (会覆盖 config 中的值)
        """
        self.config = config or FitnessConfig()

        # 使用 kwargs 更新配置
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.extra_params[key] = value

    @property
    def name(self) -> str:
        """返回适应度函数名称"""
        return self._registered_name

    @abstractmethod
    def calculate(self, eval_result: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        计算适应度值 (子类必须实现)

        Args:
            eval_result: FitnessEvaluator.evaluate() 返回的结果字典，包含:
                - spectra: 光谱响应矩阵 (16, W)
                - score_uncorrelation: 去相关性得分 (0~1)
                - score_information: 信息量得分 (log scale)
                - stability_score: 稳定性得分 (0~1)
                - condition_number: 条件数原始值
                - transmittance_score: 透过率得分 (0~1)
                - transmittance_details: 透过率详细信息
                - correlation_matrix: 相关系数矩阵

        Returns:
            Tuple[float, Dict[str, float]]:
                - fitness: 标量适应度值
                - details: 包含各分项的字典 (用于日志记录)
        """
        pass

    def __call__(self, eval_result: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """允许直接调用实例"""
        return self.calculate(eval_result)

    # =========================================================================
    # 通用惩罚计算方法 (子类可复用)
    # =========================================================================

    def _penalty_transmittance(self, trans_details: Dict) -> float:
        """透过率惩罚: 低于阈值时的平方惩罚"""
        mean_trans = trans_details['overall_mean_transmittance']
        if mean_trans < self.config.trans_threshold:
            return self.config.lambda_trans * ((self.config.trans_threshold - mean_trans) ** 2)
        return 0.0

    def _penalty_correlation(self, score_uncorrelation: float) -> float:
        """相关性惩罚: 超过阈值时的平方惩罚"""
        max_corr = 1.0 - score_uncorrelation
        if max_corr > self.config.corr_threshold:
            return self.config.lambda_corr * ((max_corr - self.config.corr_threshold) ** 2)
        return 0.0

    def _penalty_uniformity(self, trans_details: Dict) -> float:
        """均匀性惩罚: 通道间透过率标准差"""
        channel_std = trans_details['channel_std_transmittance']
        return self.config.lambda_uniformity * channel_std

    def _penalty_stability(self, cond_number: float) -> float:
        """稳定性惩罚: 带死区的合页损失"""
        log_cond = np.log10(cond_number) if cond_number > 0 else 0
        log_threshold = np.log10(self.config.cond_threshold)

        if log_cond <= log_threshold:
            return 0.0
        return self.config.lambda_stability * ((log_cond - log_threshold) ** 2)

    def _check_valid(self, eval_result: Dict) -> bool:
        """检查评估结果是否有效"""
        info = eval_result.get('score_information', -np.inf)
        return not (info == -np.inf or np.isnan(info))

    def describe(self) -> str:
        """返回适应度函数的描述信息"""
        return f"{self.name}: {self.__class__.__doc__ or '无描述'}"


# =============================================================================
# 具体适应度函数实现
# =============================================================================

@register_fitness("information")
class InformationFitness(BaseFitnessFunction):
    """
    信息量优先适应度函数

    目标: 最大化 D-Optimality (信息量)
    适用场景: 关注重建质量，希望保留尽可能多的光谱信息

    公式: Fitness = Information - P_trans - P_corr - P_unif - P_stab
    """

    def calculate(self, eval_result: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        if not self._check_valid(eval_result):
            return -1e6, {'info': 0, 'p_trans': 0, 'p_corr': 0, 'p_unif': 0, 'p_stab': 0}

        trans_details = eval_result['transmittance_details']

        # 主目标: 信息量
        reward_info = eval_result['score_information']

        # 惩罚项
        p_trans = self._penalty_transmittance(trans_details)
        p_corr = self._penalty_correlation(eval_result['score_uncorrelation'])
        p_unif = self._penalty_uniformity(trans_details)
        p_stab = self._penalty_stability(eval_result['condition_number'])

        fitness = reward_info - p_trans - p_corr - p_unif - p_stab

        details = {
            'info': reward_info,
            'p_trans': p_trans,
            'p_corr': p_corr,
            'p_unif': p_unif,
            'p_stab': p_stab
        }

        return fitness, details


@register_fitness("uncorrelation")
class UncorrelationFitness(BaseFitnessFunction):
    """
    去相关性优先适应度函数

    目标: 最小化滤光片间的相关性
    适用场景: 关注光谱分辨率，希望每个滤光片提供独立信息

    公式: Fitness = α * Uncorrelation - P_trans - P_stab
    """

    def __init__(self, config: Optional[FitnessConfig] = None,
                 alpha: float = 100.0, **kwargs):
        """
        Args:
            alpha: 去相关性得分的缩放系数 (默认100，使其与信息量量级相当)
        """
        super().__init__(config, **kwargs)
        self.alpha = alpha

    def calculate(self, eval_result: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        if not self._check_valid(eval_result):
            return -1e6, {'uncorr': 0, 'p_trans': 0, 'p_stab': 0}

        trans_details = eval_result['transmittance_details']

        # 主目标: 去相关性 (放大到合理量级)
        reward_uncorr = self.alpha * eval_result['score_uncorrelation']

        # 惩罚项
        p_trans = self._penalty_transmittance(trans_details)
        p_stab = self._penalty_stability(eval_result['condition_number'])

        fitness = reward_uncorr - p_trans - p_stab

        details = {
            'uncorr': reward_uncorr,
            'p_trans': p_trans,
            'p_stab': p_stab
        }

        return fitness, details


@register_fitness("stability")
class StabilityFitness(BaseFitnessFunction):
    """
    稳定性优先适应度函数

    目标: 最小化条件数，提高数值稳定性
    适用场景: 对噪声敏感的应用，需要鲁棒的重建

    公式: Fitness = β * Stability - P_trans - P_corr
    """

    def __init__(self, config: Optional[FitnessConfig] = None,
                 beta: float = 100.0, **kwargs):
        """
        Args:
            beta: 稳定性得分的缩放系数
        """
        super().__init__(config, **kwargs)
        self.beta = beta

    def calculate(self, eval_result: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        if not self._check_valid(eval_result):
            return -1e6, {'stability': 0, 'p_trans': 0, 'p_corr': 0}

        trans_details = eval_result['transmittance_details']

        # 主目标: 稳定性
        reward_stability = self.beta * eval_result['stability_score']

        # 惩罚项
        p_trans = self._penalty_transmittance(trans_details)
        p_corr = self._penalty_correlation(eval_result['score_uncorrelation'])

        fitness = reward_stability - p_trans - p_corr

        details = {
            'stability': reward_stability,
            'p_trans': p_trans,
            'p_corr': p_corr
        }

        return fitness, details


@register_fitness("combined")
class CombinedFitness(BaseFitnessFunction):
    """
    组合适应度函数 (默认)

    目标: 加权组合多个优化目标
    适用场景: 需要平衡多个指标的通用场景

    公式: Fitness = w1*Info + w2*α*Uncorr + w3*β*Stab - Penalties
    """

    def __init__(self, config: Optional[FitnessConfig] = None,
                 w_info: float = 1.0,
                 w_uncorr: float = 0.0,
                 w_stability: float = 0.0,
                 alpha: float = 100.0,
                 beta: float = 100.0,
                 **kwargs):
        """
        Args:
            w_info: 信息量权重
            w_uncorr: 去相关性权重
            w_stability: 稳定性权重
            alpha: 去相关性缩放系数
            beta: 稳定性缩放系数
        """
        super().__init__(config, **kwargs)
        self.w_info = w_info
        self.w_uncorr = w_uncorr
        self.w_stability = w_stability
        self.alpha = alpha
        self.beta = beta

    def calculate(self, eval_result: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        if not self._check_valid(eval_result):
            return -1e6, {'info': 0, 'uncorr': 0, 'stability': 0,
                         'p_trans': 0, 'p_corr': 0, 'p_unif': 0, 'p_stab': 0}

        trans_details = eval_result['transmittance_details']

        # 多目标加权
        reward_info = self.w_info * eval_result['score_information']
        reward_uncorr = self.w_uncorr * self.alpha * eval_result['score_uncorrelation']
        reward_stability = self.w_stability * self.beta * eval_result['stability_score']

        total_reward = reward_info + reward_uncorr + reward_stability

        # 惩罚项
        p_trans = self._penalty_transmittance(trans_details)
        p_corr = self._penalty_correlation(eval_result['score_uncorrelation'])
        p_unif = self._penalty_uniformity(trans_details)
        p_stab = self._penalty_stability(eval_result['condition_number'])

        fitness = total_reward - p_trans - p_corr - p_unif - p_stab

        details = {
            'info': reward_info,
            'uncorr': reward_uncorr,
            'stability': reward_stability,
            'p_trans': p_trans,
            'p_corr': p_corr,
            'p_unif': p_unif,
            'p_stab': p_stab
        }

        return fitness, details


@register_fitness("weighted_sum")
class WeightedSumFitness(BaseFitnessFunction):
    """
    简单加权和适应度函数

    目标: 使用归一化指标的简单加权和
    适用场景: 需要各指标贡献可解释的场景

    公式: Fitness = w1*Info_norm + w2*Uncorr + w3*Stability + w4*Trans
    """

    def __init__(self, config: Optional[FitnessConfig] = None,
                 w_info: float = 0.4,
                 w_uncorr: float = 0.3,
                 w_stability: float = 0.2,
                 w_trans: float = 0.1,
                 info_range: Tuple[float, float] = (50.0, 150.0),
                 **kwargs):
        """
        Args:
            w_info, w_uncorr, w_stability, w_trans: 各指标权重 (建议和为1)
            info_range: 信息量归一化范围 [min, max]
        """
        super().__init__(config, **kwargs)
        self.w_info = w_info
        self.w_uncorr = w_uncorr
        self.w_stability = w_stability
        self.w_trans = w_trans
        self.info_min, self.info_max = info_range

    def _normalize_info(self, info: float) -> float:
        """将信息量归一化到 [0, 1]"""
        return np.clip((info - self.info_min) / (self.info_max - self.info_min), 0, 1)

    def calculate(self, eval_result: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        if not self._check_valid(eval_result):
            return 0.0, {'info_norm': 0, 'uncorr': 0, 'stability': 0, 'trans': 0}

        trans_details = eval_result['transmittance_details']

        # 归一化各指标到 [0, 1]
        info_norm = self._normalize_info(eval_result['score_information'])
        uncorr = eval_result['score_uncorrelation']
        stability = eval_result['stability_score']
        trans = trans_details['overall_mean_transmittance']

        # 加权求和
        fitness = (self.w_info * info_norm +
                   self.w_uncorr * uncorr +
                   self.w_stability * stability +
                   self.w_trans * trans)

        details = {
            'info_norm': info_norm,
            'uncorr': uncorr,
            'stability': stability,
            'trans': trans
        }

        return fitness, details


@register_fitness("pareto")
class ParetoFitness(BaseFitnessFunction):
    """
    Pareto 前沿适应度函数

    目标: 基于 Pareto 支配关系的适应度
    适用场景: 多目标优化，需要保留多样性的 Pareto 前沿

    注意: 此函数返回的是单个体的目标向量范数，完整的 Pareto 排序需要在种群级别实现
    """

    def __init__(self, config: Optional[FitnessConfig] = None,
                 objectives: List[str] = None,
                 **kwargs):
        """
        Args:
            objectives: 优化目标列表，可选:
                'info', 'uncorr', 'stability', 'trans', 'cond'
        """
        super().__init__(config, **kwargs)
        self.objectives = objectives or ['info', 'uncorr', 'stability']

    def calculate(self, eval_result: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        if not self._check_valid(eval_result):
            return -1e6, {obj: 0 for obj in self.objectives}

        trans_details = eval_result['transmittance_details']

        # 提取各目标值
        obj_values = {
            'info': eval_result['score_information'],
            'uncorr': eval_result['score_uncorrelation'],
            'stability': eval_result['stability_score'],
            'trans': trans_details['overall_mean_transmittance'],
            'cond': -np.log10(eval_result['condition_number'])  # 取负数使其越大越好
        }

        # 计算选定目标的范数作为适应度
        selected_values = [obj_values[obj] for obj in self.objectives if obj in obj_values]
        fitness = np.linalg.norm(selected_values)

        details = {obj: obj_values.get(obj, 0) for obj in self.objectives}

        return fitness, details


# =============================================================================
# 自定义适应度函数模板
# =============================================================================

# @register_fitness("custom")
# class CustomFitness(BaseFitnessFunction):
#     """
#     自定义适应度函数模板
#
#     复制此模板并修改 calculate() 方法来创建新的适应度函数
#     """
#
#     def __init__(self, config: Optional[FitnessConfig] = None, **kwargs):
#         super().__init__(config, **kwargs)
#         # 在这里添加自定义参数
#
#     def calculate(self, eval_result: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
#         if not self._check_valid(eval_result):
#             return -1e6, {}
#
#         # 在这里实现自定义的适应度计算逻辑
#         fitness = 0.0
#         details = {}
#
#         return fitness, details


# =============================================================================
# 工具函数
# =============================================================================

def print_fitness_functions():
    """打印所有可用的适应度函数及其描述"""
    print("\n" + "=" * 60)
    print("可用的适应度函数")
    print("=" * 60)

    for name, cls in _FITNESS_REGISTRY.items():
        doc = cls.__doc__ or "无描述"
        # 提取第一行作为简短描述
        short_desc = doc.strip().split('\n')[0]
        print(f"\n  [{name}]")
        print(f"    {short_desc}")

    print("\n" + "=" * 60)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 打印可用的适应度函数
    print_fitness_functions()

    # 测试各适应度函数
    print("\n测试适应度函数...")

    # 模拟评估结果
    mock_result = {
        'spectra': np.random.rand(16, 301),
        'score_uncorrelation': 0.75,
        'score_information': 95.5,
        'stability_score': 0.85,
        'condition_number': 50.0,
        'transmittance_score': 0.8,
        'transmittance_details': {
            'overall_mean_transmittance': 0.6,
            'channel_std_transmittance': 0.1,
            'channel_mean_transmittance': np.random.rand(16) * 0.3 + 0.5,
            'mean_score': 0.9,
            'uniformity_score': 0.8
        },
        'correlation_matrix': np.eye(16)
    }

    for name in list_fitness_functions():
        fitness_fn = get_fitness_function(name)
        fitness, details = fitness_fn(mock_result)
        print(f"\n  {name}: fitness = {fitness:.4f}")
        print(f"    details: {details}")
