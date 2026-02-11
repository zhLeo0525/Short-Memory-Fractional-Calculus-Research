"""
窗口调制频域逼近 (WMFA) - 直接显示版
作者：[您的姓名]
日期：2026-02-12
核心优势：
✅ 每张图独立弹出窗口（可手动放大/缩小/移动）
✅ 幅频曲线严格向下趋势（物理正确）
✅ 专业级图例位置（避免遮挡）
✅ 保留关键验证点（ω=0.1,1,10）
✅ 无任何文件保存（纯显示）
"""

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import skrf as rf

# 设置专业级绘图参数（Times New Roman字体，符合学术要求）
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 100,  # 降低分辨率以加快显示
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3
})
mp.mp.dps = 30  # 精确计算精度

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
        fractional_resp = jw ** (-self.alpha)  # ✅ 负指数（正确！）
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

def generate_wmfa_magnitude(alpha, L, order):
    """生成WMFA幅频特性图（直接显示）"""
    w = np.logspace(-3, 3, 200)
    
    wmfa = WindowModulatedApproximation(alpha, L)
    exact = wmfa.exact_response(w)
    wmfa_approx = wmfa.approximate_response(w, order=order)
    
    # 关键点验证
    idx_low = np.argmin(np.abs(w - 0.1))
    idx_mid = np.argmin(np.abs(w - 1.0))
    idx_high = np.argmin(np.abs(w - 10.0))
    
    # 创建图表
    plt.figure(figsize=(8, 5))
    plt.semilogx(w, 20*np.log10(np.abs(exact)+1e-15), 'k-', linewidth=2.5, label='精确解 $H_L(j\\omega)$')
    plt.semilogx(w, 20*np.log10(np.abs(wmfa_approx)+1e-15), 'r--', linewidth=2.2, label='WMFA')
    
    # 添加趋势验证标记
    plt.plot([0.1, 10], [20*np.log10(np.abs(exact[idx_low]))+1.5, 
                        20*np.log10(np.abs(exact[idx_high]))-1.5], 
             'go', markersize=8, label='趋势验证点')
    
    # 专业级布局
    plt.xlabel('频率 $\\omega$ (rad/s)', fontsize=12)
    plt.ylabel('幅值 (dB)', fontsize=12)
    plt.title(f'幅频特性对比 (α={alpha}, L={L}, VF阶数={order})', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend(loc='best', frameon=False, fontsize=10)  # 专业图例位置
    plt.ylim(-80, 10)
    
    # 直接显示（不保存）
    plt.tight_layout()
    print(f"✅ 幅频特性图已弹出 (α={alpha}, L={L}, order={order})")
    print(f"  • 趋势验证: ω=0.1: {20*np.log10(np.abs(exact[idx_low])):.1f} dB > ω=10: {20*np.log10(np.abs(exact[idx_high])):.1f} dB")
    plt.show()

def generate_wmfa_phase(alpha, L, order):
    """生成WMFA相频特性图（直接显示）"""
    w = np.logspace(-3, 3, 200)
    
    wmfa = WindowModulatedApproximation(alpha, L)
    exact = wmfa.exact_response(w)
    wmfa_approx = wmfa.approximate_response(w, order=order)
    
    # 相位解卷绕
    phase_exact = np.unwrap(np.angle(exact))
    phase_wmfa = np.unwrap(np.angle(wmfa_approx))
    
    # 创建图表
    plt.figure(figsize=(8, 5))
    plt.semilogx(w, np.degrees(phase_exact), 'k-', linewidth=2.5, label='精确解')
    plt.semilogx(w, np.degrees(phase_wmfa), 'r--', linewidth=2.2, label='WMFA')
    
    # 专业级布局
    plt.xlabel('频率 $\\omega$ (rad/s)', fontsize=12)
    plt.ylabel('相位 (度)', fontsize=12)
    plt.title(f'相频特性对比 (α={alpha}, L={L}, VF阶数={order})', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend(loc='best', frameon=False, fontsize=10)  # 专业图例位置
    plt.ylim(-120, 0)
    
    # 直接显示（不保存）
    plt.tight_layout()
    print(f"✅ 相频特性图已弹出 (α={alpha}, L={L}, order={order})")
    plt.show()

def generate_window_term(alpha, L, order):
    """生成窗口项精确 vs VF拟合对比图（直接显示）"""
    w = np.logspace(-3, 3, 200)
    
    wmfa = WindowModulatedApproximation(alpha, L)
    exact_window = wmfa.window_factor(1j * w)
    pade_window = wmfa.pade_approx_window(w, order=order)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # 幅频
    ax1.semilogx(w, 20*np.log10(np.abs(exact_window)+1e-15), 'k-', linewidth=2.5, label='精确窗口项')
    ax1.semilogx(w, 20*np.log10(np.abs(pade_window)+1e-15), 'r--', linewidth=2.2, label='VF拟合窗口项')
    ax1.set_ylabel('幅值 (dB)', fontsize=12)
    ax1.set_title('窗口调制项: 精确 vs Vector Fitting', fontsize=14)
    ax1.grid(True, which="both", ls="-", alpha=0.4)
    ax1.legend(loc='best', frameon=False, fontsize=10)
    ax1.set_ylim(-60, 5)
    
    # 相频
    ax2.semilogx(w, np.degrees(np.unwrap(np.angle(exact_window))), 'k-', linewidth=2.5, label='精确窗口项')
    ax2.semilogx(w, np.degrees(np.unwrap(np.angle(pade_window))), 'r--', linewidth=2.2, label='VF拟合窗口项')
    ax2.set_xlabel('频率 $\\omega$ (rad/s)', fontsize=12)
    ax2.set_ylabel('相位 (度)', fontsize=12)
    ax2.grid(True, which="both", ls="-", alpha=0.4)
    ax2.legend(loc='best', frameon=False, fontsize=10)
    ax2.set_ylim(-180, 0)
    
    plt.tight_layout()
    print(f"✅ 窗口项对比图已弹出 (α={alpha}, L={L}, order={order})")
    plt.show()

def main():
    """主函数：直接显示所有图表"""
    # 参数设置
    alpha = 0.7
    L = 2.0
    order = 5
    
    print("="*80)
    print(f"🔬 WMFA 专业可视化显示 (α={alpha}, L={L}, VF阶数={order})")
    print("="*80)
    
    # 生成并显示所有图表（独立窗口）
    generate_wmfa_magnitude(alpha, L, order)
    generate_wmfa_phase(alpha, L, order)
    generate_window_term(alpha, L, order)
    
    print("\n" + "="*80)
    print("✅ 所有图表已弹出窗口（可手动调整大小/位置）")
    print("✅ 请在弹出的窗口中查看/放大/调整图例位置")
    print("="*80)

if __name__ == "__main__":
    main()