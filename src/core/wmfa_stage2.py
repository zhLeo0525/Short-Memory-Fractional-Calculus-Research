# === 添加以下函数到文件顶部 ===
import math
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import skrf as rf
mpl.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
mpl.rcParams["axes.unicode_minus"] = False
class WindowModulatedApproximation:
    """窗口调制频域逼近 (WMFA) 实现"""
    def __init__(self, alpha: float, L: float):
        assert 0 < alpha < 1, "阶数alpha必须在(0,1)区间"
        assert L > 0, "记忆窗口长度L必须大于0"
        self.alpha = alpha
        self.L = L
        self.omega_b = 1e-3
        self.omega_h = 1e3
        self._vf_cache = None
        self._vf_cache_key = None

    def window_factor(self, s: np.ndarray) -> np.ndarray:
        """计算窗口调制项 γ(1-α, sL)/Γ(1-α)"""
        s_arr = np.asarray(s, dtype=complex)
        z_values = s_arr * self.L
        flattened = z_values.ravel()
        values = []
        for z in flattened:
            mp_z = mp.mpc(z.real, z.imag)
            regularized = mp.gammainc(1 - self.alpha, 0, mp_z, regularized=True)
            values.append(complex(regularized))
        return np.array(values, dtype=complex).reshape(z_values.shape)
    
    def exact_response(self, w: np.ndarray) -> np.ndarray:
        """计算精确频率响应 H_L(jω) = [γ(1-α, jωL)/Γ(1-α)] * (jω)^{-α}"""
        jw = 1j * w
        window_resp = self.window_factor(jw)
        fractional_resp = jw ** (-self.alpha)  # ✅ 负指数（正确！��
        return window_resp * fractional_resp
    
    def pade_approx_window(self, w: np.ndarray, order: int = 5) -> np.ndarray:
        """使用Vector Fitting进行窗口项拟合"""
        w_grid = np.logspace(np.log10(self.omega_b), np.log10(self.omega_h), 80)
        s_grid = 1j * w_grid
        exact_window = self.window_factor(s_grid)

        freq_grid = rf.Frequency.from_f(w_grid / (2 * np.pi), unit="hz")
        s_params = exact_window.reshape(-1, 1, 1)
        network = rf.Network(frequency=freq_grid, s=s_params)

        cache_key = (self.alpha, self.L, order)
        if self._vf_cache_key == cache_key and self._vf_cache is not None:
            vf = self._vf_cache
        else:
            vf = rf.vectorFitting.VectorFitting(network)
            n_poles_real = max(1, order // 2)
            n_poles_cmplx = max(1, order - n_poles_real)
            
            # 安全调用VectorFitting
            try:
                vf.vector_fit(
                    n_poles_real=n_poles_real,
                    n_poles_cmplx=n_poles_cmplx,
                    init_pole_spacing="log",
                    parameter_type="s",
                    fit_constant=True,
                    fit_proportional=False
                )
            except TypeError:
                vf.vector_fit(
                    n_poles_real=n_poles_real,
                    n_poles_cmplx=n_poles_cmplx,
                    fit_constant=True,
                    fit_proportional=False
                )
            
            self._vf_cache = vf
            self._vf_cache_key = cache_key

        freq_eval = w / (2 * np.pi)
        return vf.get_model_response(0, 0, freqs=freq_eval)
    
    def approximate_response(self, w: np.ndarray, order: int = 5) -> np.ndarray:
        """WMFA近似频率响应"""
        pade_window = self.pade_approx_window(w, order)
        fractional_resp = (1j * w) ** (-self.alpha)
        return pade_window * fractional_resp

    def oustaloup_approx(self, w: np.ndarray, order: int = 5) -> np.ndarray:
        """标准Oustaloup方法逼近 s^{-α} (2N+1零极点)"""
        omega_b, omega_h = self.omega_b, self.omega_h
        N = order

        zeros = []
        poles = []
        for k in range(2 * N + 1):
            omega_z = omega_b * (omega_h / omega_b) ** ((k + (1 - self.alpha) / 2) / (2 * N + 1))
            omega_p = omega_b * (omega_h / omega_b) ** ((k + (1 + self.alpha) / 2) / (2 * N + 1))
            zeros.append(omega_z)
            poles.append(omega_p)

        gain = omega_h ** (-self.alpha)
        s = 1j * w[:, None]
        num = np.prod(s / np.array(zeros) + 1, axis=1)
        den = np.prod(s / np.array(poles) + 1, axis=1)
        response = gain * num / den

        # Mid-band normalization
        omega_ref = np.sqrt(omega_b * omega_h)
        s_ref = 1j * omega_ref
        num_ref = np.prod(s_ref / np.array(zeros) + 1)
        den_ref = np.prod(s_ref / np.array(poles) + 1)
        approx_ref = gain * num_ref / den_ref
        exact_ref = s_ref ** (-self.alpha)
        return response * (exact_ref / approx_ref)

    
    def calculate_error(self, exact, approx):
        """计算相对误差"""
        return np.abs(exact - approx) / np.maximum(np.abs(exact), np.abs(approx))

# === Stage 2函数 ===
def parameter_sensitivity_analysis():
    """生成L-order误差热力图 (α=0.7固定)"""
    L_vals = np.linspace(1.0, 3.0, 10)  # L: 1.0 to 3.0
    order_vals = np.arange(3, 9)         # VF阶数: 3 to 8
    error_matrix = np.zeros((len(L_vals), len(order_vals)))
    
    for i, L in enumerate(L_vals):
        for j, order in enumerate(order_vals):
            wmfa = WindowModulatedApproximation(alpha=0.7, L=L)
            w = np.logspace(-3, 3, 200)
            exact = wmfa.exact_response(w)
            approx = wmfa.approximate_response(w, order=order)
            error_matrix[i, j] = np.max(wmfa.calculate_error(exact, approx))
    
    # 绘制热力图
    plt.figure(figsize=(10, 6))
    im = plt.imshow(error_matrix*100, cmap='viridis', 
                    extent=[order_vals[0]-0.5, order_vals[-1]+0.5, 
                            L_vals[-1], L_vals[0]], aspect='auto')
    plt.colorbar(im, label='最大相对误差 (%)')
    plt.xlabel('VF阶数', fontsize=12)
    plt.ylabel('记忆窗口长度 L', fontsize=12)
    plt.title('WMFA参数敏感性分析 (α=0.7)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/figures/parameter_sensitivity.png', dpi=300)
    plt.show()
    
    # 输出最优参数
    min_idx = np.unravel_index(np.argmin(error_matrix), error_matrix.shape)
    L_opt = L_vals[min_idx[0]]
    order_opt = order_vals[min_idx[1]]
    print(f"\n✅ 最优参数组合: L={L_opt:.2f}, VF阶数={order_opt}, 误差={np.min(error_matrix)*100:.2f}%")
    return L_opt, order_opt

def compare_with_oustaloup(alpha, L, order):
    """WMFA vs Oustaloup误差对比 (短记忆场景)"""
    w = np.logspace(-3, 3, 200)
    wmfa = WindowModulatedApproximation(alpha, L)
    
    # WMFA结果
    wmfa_resp = wmfa.approximate_response(w, order=order)
    
    # Oustaloup结果 (内部实现，避免外部依赖)
    oustaloup_resp = wmfa.oustaloup_approx(w, order=order)
    
    # 计算误差
    exact = wmfa.exact_response(w)
    wmfa_error = wmfa.calculate_error(exact, wmfa_resp)
    oustaloup_error = wmfa.calculate_error(exact, oustaloup_resp)
    
    # 绘制对比
    plt.figure(figsize=(8, 5))
    plt.semilogx(w, wmfa_error, 'r-', linewidth=2.2, label='WMFA')
    plt.semilogx(w, oustaloup_error, 'b--', linewidth=2.2, label='Oustaloup')
    plt.axhline(0.01, color='k', linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.title(f'误差对比: WMFA vs Oustaloup (α={alpha}, L={L})', fontsize=14)
    plt.xlabel('频率 $\\omega$ (rad/s)', fontsize=12)
    plt.ylabel('相对误差', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('results/figures/error_comparison.png', dpi=300)
    plt.show()
    
    # 输出关键指标
    wmfa_max_err = np.max(wmfa_error) * 100
    oustaloup_max_err = np.max(oustaloup_error) * 100
    print(f"\nWMFA最大误差: {wmfa_max_err:.2f}%")
    print(f"Oustaloup最大误差: {oustaloup_max_err:.2f}%")
    print(f"WMFA比Oustaloup误差低: {((oustaloup_max_err - wmfa_max_err)/oustaloup_max_err)*100:.1f}%")
    
    # 生成统计表
    print("\n📊 误差统计表:")
    print(f"{'Method':<15} | {'Max Error (%)':<12} | '1% Error Band (%)'")
    print("-"*50)
    print(f"{'WMFA':<15} | {wmfa_max_err:<12.2f} | {np.sum(wmfa_error < 0.01)/len(wmfa_error)*100:.1f}")
    print(f"{'Oustaloup':<15} | {oustaloup_max_err:<12.2f} | {np.sum(oustaloup_error < 0.01)/len(oustaloup_error)*100:.1f}")
    
    return wmfa_max_err, oustaloup_max_err

def generate_wmfa_magnitude(alpha, L, order):
    """幅频特性图 (Stage 1)"""
    w = np.logspace(-3, 3, 200)
    wmfa = WindowModulatedApproximation(alpha, L)
    exact = wmfa.exact_response(w)
    approx = wmfa.approximate_response(w, order=order)
    
    plt.figure(figsize=(8, 5))
    plt.semilogx(w, 20*np.log10(np.abs(exact)), 'k-', linewidth=2.0, label='Exact')
    plt.semilogx(w, 20*np.log10(np.abs(approx)), 'r--', linewidth=2.0, label='WMFA')
    
    # 标记关键点
    idx_01 = np.argmin(np.abs(w - 0.1))
    idx_10 = np.argmin(np.abs(w - 10))
    plt.plot(w[idx_01], 20*np.log10(np.abs(exact[idx_01])), 'go', markersize=8)
    plt.plot(w[idx_10], 20*np.log10(np.abs(exact[idx_10])), 'go', markersize=8)
    
    plt.title(f'幅频特性: α={alpha}, L={L}, VF阶数={order}', fontsize=14)
    plt.xlabel('频率 $\\omega$ (rad/s)', fontsize=12)
    plt.ylabel('幅值 (dB)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'results/figures/wmfa_magnitude_alpha{alpha}_L{L}_order{order}.png', dpi=300)
    plt.show()

def generate_wmfa_phase(alpha, L, order):
    """相频特性图 (Stage 1)"""
    w = np.logspace(-3, 3, 200)
    wmfa = WindowModulatedApproximation(alpha, L)
    exact = wmfa.exact_response(w)
    approx = wmfa.approximate_response(w, order=order)
    
    plt.figure(figsize=(8, 5))
    plt.semilogx(w, np.angle(exact, deg=True), 'k-', linewidth=2.0, label='Exact')
    plt.semilogx(w, np.angle(approx, deg=True), 'r--', linewidth=2.0, label='WMFA')
    
    # 标记关键点
    idx_01 = np.argmin(np.abs(w - 0.1))
    idx_10 = np.argmin(np.abs(w - 10))
    plt.plot(w[idx_01], np.angle(exact[idx_01], deg=True), 'go', markersize=8)
    plt.plot(w[idx_10], np.angle(exact[idx_10], deg=True), 'go', markersize=8)
    
    plt.title(f'相频特性: α={alpha}, L={L}, VF阶数={order}', fontsize=14)
    plt.xlabel('频率 $\\omega$ (rad/s)', fontsize=12)
    plt.ylabel('相位 (°)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'results/figures/wmfa_phase_alpha{alpha}_L{L}_order{order}.png', dpi=300)
    plt.show()

def generate_window_term(alpha, L, order):
    """窗口项对比图 (Stage 1)"""
    w = np.logspace(-3, 3, 200)
    wmfa = WindowModulatedApproximation(alpha, L)
    
    # 提取窗口项
    window_exact = wmfa.exact_response(w) / (1j * w)**(-alpha)
    window_approx = wmfa.approximate_response(w, order=order) / (1j * w)**(-alpha)
    
    plt.figure(figsize=(8, 5))
    plt.semilogx(w, 20*np.log10(np.abs(window_exact)), 'k-', linewidth=2.0, label='Exact')
    plt.semilogx(w, 20*np.log10(np.abs(window_approx)), 'r--', linewidth=2.0, label='WMFA')
    
    plt.title(f'窗口项: α={alpha}, L={L}, VF阶数={order}', fontsize=14)
    plt.xlabel('频率 $\\omega$ (rad/s)', fontsize=12)
    plt.ylabel('幅值 (dB)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'results/figures/window_term_alpha{alpha}_L{L}_order{order}.png', dpi=300)
    plt.show()

# === 主函数 ===
def main():
    # 设置参数
    alpha = 0.7
    L = 2.0
    order = 5
    
    # Stage 1: 物理验证
    print("="*60)
    print("Stage 1: 物理特性验证")
    print("="*60)
    generate_wmfa_magnitude(alpha, L, order)
    generate_wmfa_phase(alpha, L, order)
    generate_window_term(alpha, L, order)
    
    # Stage 2: 多参数分析
    print("\n" + "="*60)
    print("Stage 2: 多参数敏感性分析")
    print("="*60)
    L_opt, order_opt = parameter_sensitivity_analysis()
    
    print("\n" + "="*60)
    print("Stage 2: WMFA vs Oustaloup对比")
    print("="*60)
    compare_with_oustaloup(alpha, L, order)
    
    print("\n" + "="*60)
    print("Stage 2 完成: 所有关键验证数据已生成")
    print("="*60)

if __name__ == "__main__":
    main()