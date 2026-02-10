"""
短记忆分数阶算子核心实现
作者：[您的姓名]
日期：2026-02-10
"""

import numpy as np
from scipy.special import gamma, gammainc, gammaincc
from scipy.integrate import quad
from typing import Callable, Tuple


class ShortMemoryFractionalOperator:
    """
    短记忆分数阶积分/微分算子实现
    支持：Caputo型短记忆分数阶积分（阶数α∈(0,1)）
    """

    def __init__(self, alpha: float, L: float, t0: float = 0.0):
        """
        初始化短记忆算子

        参数:
        alpha : float - 分数阶阶数 (0 < alpha < 1)
        L     : float - 记忆窗口长度 (L > 0)
        t0    : float - 初始时间 (默认0)
        """
        assert 0 < alpha < 1, "阶数alpha必须在(0,1)区间"
        assert L > 0, "记忆窗口长度L必须大于0"

        self.alpha = alpha
        self.L = L
        self.t0 = t0
        self.gamma_alpha = gamma(alpha)

    def _integration_lower_bound(self, t: float) -> float:
        """
        计算积分下限 s_L(t)
        s_L(t) = t0, 当 t < t0 + L
                = t - L, 当 t >= t0 + L
        """
        return self.t0 if t < self.t0 + self.L else t - self.L

    def kernel(self, t: float, tau: float) -> float:
        """
        短记忆核函数: k(t,τ) = (t-τ)^(α-1)/Γ(α) * I_{[s_L(t), t]}(τ)

        参数:
        t   : 当前时间
        tau : 积分变量

        返回:
        核函数值
        """
        if tau < self._integration_lower_bound(t) or tau > t:
            return 0.0
        return (t - tau) ** (self.alpha - 1) / self.gamma_alpha

    def compute_integral(self, f: Callable, t: float) -> float:
        """
        计算短记忆分数阶积分: I_{s_L(t)}^α f(t)

        参数:
        f : 被积函数 (Callable: tau -> f(tau))
        t : 当前时间点

        返回:
        积分值
        """
        lower = self._integration_lower_bound(t)

        # 使用自适应积分（处理奇异性）
        result, error = quad(
            lambda tau: self.kernel(t, tau) * f(tau),
            lower,
            t,
            epsabs=1e-8,
            epsrel=1e-6,
            limit=100,
        )
        return result

    def compute_array(self, f: Callable, t_array: np.ndarray) -> np.ndarray:
        """
        批量计算短记忆分数阶积分（向量化）

        参数:
        f       : 被积函数
        t_array : 时间数组

        返回:
        积分结果数组
        """
        results = np.zeros_like(t_array, dtype=float)
        for i, t in enumerate(t_array):
            results[i] = self.compute_integral(f, t)
        return results


class InfiniteMemoryFractionalOperator:
    """
    无限记忆分数阶积分器（用于对比）
    使用标准Riemann-Liouville定义
    """

    def __init__(self, alpha: float, t0: float = 0.0):
        self.alpha = alpha
        self.t0 = t0
        self.gamma_alpha = gamma(alpha)

    def kernel(self, t: float, tau: float) -> float:
        """无限记忆核函数: (t-τ)^(α-1)/Γ(α)"""
        if tau < self.t0 or tau > t:
            return 0.0
        return (t - tau) ** (self.alpha - 1) / self.gamma_alpha

    def compute_array(self, f: Callable, t_array: np.ndarray) -> np.ndarray:
        """批量计算（使用数值积分）"""
        results = np.zeros_like(t_array, dtype=float)
        for i, t in enumerate(t_array):
            result, _ = quad(
                lambda tau: self.kernel(t, tau) * f(tau),
                self.t0,
                t,
                epsabs=1e-8,
                epsrel=1e-6,
                limit=200,
            )
            results[i] = result
        return results


def analytical_solution_step(alpha: float, t: np.ndarray, L: float = None) -> np.ndarray:
    """
    阶跃输入(单位阶跃)下分数阶积分的解析解

    无限记忆: I^α 1 = t^α / Γ(α+1)
    短记忆:   I_{s_L(t)}^α 1 = [t^α - (t-L)_+^α] / Γ(α+1)

    参数:
    alpha : 阶数
    t     : 时间数组
    L     : 记忆窗口长度 (None表示无限记忆)

    返回:
    解析解数组
    """
    gamma_alpha1 = gamma(alpha + 1)

    if L is None:
        return t ** alpha / gamma_alpha1

    result = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < L:
            result[i] = ti ** alpha / gamma_alpha1
        else:
            result[i] = (ti ** alpha - (ti - L) ** alpha) / gamma_alpha1
    return result
