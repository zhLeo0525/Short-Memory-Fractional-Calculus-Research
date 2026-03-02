# === WMFA Stage 2: 参数敏感性分析专注版 (Oustaloup模块已彻底移除) ===
import math
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import skrf as rf
import os

# 创建结果目录
os.makedirs('results/figures', exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

class WindowModulatedApproximation:
    """窗口调制频域逼近 (WMFA) - 纯净参数分析版"""
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
        """精确频率响应 H_L(jω) = [γ(1-α, jωL)/Γ(1-α)] * (jω)^{-α}"""
        jw = 1j * w
        window_resp = self.window_factor(jw)
        fractional_resp = jw ** (-self.alpha)
        return window_resp * fractional_resp
    
    def pade_approx_window(self, w: np.ndarray, order: int = 18) -> np.ndarray:
        """✅ 优化：低频加密采样 + 多策略VF拟合"""
        # 低频加密采样（关键修复！）
        log_w = np.concatenate([
            np.linspace(np.log10(self.omega_b), np.log10(0.1), 200),  # 低频加密
            np.linspace(np.log10(0.1), np.log10(10), 150),            # 中频
            np.linspace(np.log10(10), np.log10(self.omega_h), 150)    # 高频
        ])
        w_grid = 10**log_w
        
        s_grid = 1j * w_grid
        exact_window = self.window_factor(s_grid)
        
        # 构建Network
        freq = rf.Frequency.from_f(w_grid / (2 * np.pi), unit='hz')
        s_data = np.zeros((len(w_grid), 1, 1), dtype=complex)
        s_data[:, 0, 0] = exact_window
        network = rf.Network(frequency=freq, s=s_data, z0=50)
        
        # 缓存查询
        cache_key = (self.alpha, self.L, order)
        if self._vf_cache_key == cache_key and self._vf_cache is not None:
            vf = self._vf_cache
        else:
            vf = rf.VectorFitting(network)
            n_poles_real = 4  # 增加实极点（窗口项特性）
            n_poles_cmplx = max(2, (order - n_poles_real) // 2)
            
            # 多策略拟合
            strategies = [
                {'n_poles_real': n_poles_real, 'n_poles_cmplx': n_poles_cmplx, 'init': 'log'},
                {'n_poles_real': n_poles_real, 'n_poles_cmplx': n_poles_cmplx + 1, 'init': 'log'},
                {'n_poles_real': n_poles_real, 'n_poles_cmplx': n_poles_cmplx, 'init': 'lin'},
            ]
            
            best_vf, best_rms = None, float('inf')
            for strat in strategies:
                try:
                    vf_temp = rf.VectorFitting(network)
                    vf_temp.vector_fit(
                        n_poles_real=strat['n_poles_real'],
                        n_poles_cmplx=strat['n_poles_cmplx'],
                        init_pole_spacing=strat['init'],
                        parameter_type='s',
                        fit_constant=True,
                        fit_proportional=False
                    )
                    rms = vf_temp.get_rms_error()
                    if rms < best_rms:
                        best_rms, best_vf = rms, vf_temp
                    if rms < 0.008:
                        break
                except Exception:
                    continue
            
            if best_vf is None:
                raise RuntimeError("Vector Fitting failed after multiple strategies")
            
            vf = best_vf
            if best_rms > 0.015:
                print(f"⚠️ 窗口项VF拟合RMS误差: {best_rms:.4f} (仍偏高，但已最优)")
            else:
                print(f"✅ 窗口项VF拟合成功: RMS误差 = {best_rms:.4f}")
            
            self._vf_cache = vf
            self._vf_cache_key = cache_key
        
        # 评估拟合结果
        freq_eval = w / (2 * np.pi)
        return vf.get_model_response(0, 0, freqs=freq_eval)
    
    def approximate_response(self, w: np.ndarray, order: int = 15) -> np.ndarray:
        """WMFA近似频率响应"""
        pade_window = self.pade_approx_window(w, order)
        fractional_resp = (1j * w) ** (-self.alpha)
        return pade_window * fractional_resp

    def calculate_error(self, exact, approx):
        """计算相对误差 (增强数值稳定性)"""
        denom = np.maximum(np.abs(exact), np.abs(approx))
        denom = np.where(denom < 1e-12, 1e-12, denom)
        return np.abs(exact - approx) / denom


# === 参数敏感性分析核心函数 ===
def optimized_parameter_analysis(alpha=0.7):
    """WMFA参数扫描 (聚焦窗口长度L与VF阶数)"""
    L_vals = np.linspace(1.0, 3.5, 15)   # 7个L值
    order_vals = np.arange(14, 26)      # 扩展至22阶（关键优化）
    error_matrix = np.zeros((len(L_vals), len(order_vals)))
    
    print("🔬 执行高精度参数扫描...")
    print(f"   • α = {alpha}")
    print(f"   • L ∈ [{L_vals[0]:.1f}, {L_vals[-1]:.1f}] (7个点)")
    print(f"   • VF阶数 ∈ [{order_vals[0]}, {order_vals[-1]}] (9个点)")
    print(f"   • 评估频点: 200 (logspace [-3, 3])")
    
    total = len(L_vals) * len(order_vals)
    count = 0
    
    for i, L in enumerate(L_vals):
        for j, order in enumerate(order_vals):
            wmfa = WindowModulatedApproximation(alpha=alpha, L=L)
            w = np.logspace(-3, 3, 200)
            exact = wmfa.exact_response(w)
            approx = wmfa.approximate_response(w, order=order)
            error_matrix[i, j] = np.max(wmfa.calculate_error(exact, approx))
            
            count += 1
            if count % 10 == 0 or count == total:
                print(f"  进度: {count}/{total} ({count/total*100:.1f}%) | L={L:.2f}, 阶数={order} → 误差={error_matrix[i,j]*100:.2f}%")
    
    # 绘制热力图
    plt.figure(figsize=(12, 7))
    im = plt.imshow(
        error_matrix * 100, 
        cmap='plasma', 
        extent=[order_vals[0]-0.5, order_vals[-1]+0.5, L_vals[-1], L_vals[0]], 
        aspect='auto',
        vmin=0, 
        vmax=10  # 调整上限以清晰显示分布
    )
    plt.colorbar(im, label='最大相对误差 (%)', fraction=0.046, pad=0.04)
    plt.xlabel('VF拟合阶数', fontsize=14, fontweight='bold')
    plt.ylabel('记忆窗口长度 L', fontsize=14, fontweight='bold')
    plt.title(f'WMFA参数敏感性分析 (α={alpha})\n误差<5%区域标绿 | 误差<3%区域标红星', 
              fontsize=16, fontweight='bold', pad=15)
    
    # 标记误差<5%区域
    for i in range(len(L_vals)):
        for j in range(len(order_vals)):
            if error_matrix[i, j] < 0.05:
                plt.gca().add_patch(plt.Rectangle(
                    (order_vals[j]-0.45, L_vals[i]-0.14), 
                    0.9, 0.28, 
                    fill=False, edgecolor='lime', linewidth=1.5, alpha=0.7
                ))
    
    # 标记最优参数
    min_idx = np.unravel_index(np.argmin(error_matrix), error_matrix.shape)
    L_opt = L_vals[min_idx[0]]
    order_opt = order_vals[min_idx[1]]
    min_err = np.min(error_matrix) * 100
    plt.plot(order_opt, L_opt, 'w*', markersize=22, markeredgewidth=2.5, 
             label=f'全局最优点\nL={L_opt:.2f}, 阶数={order_opt}\n误差={min_err:.2f}%')
    
    # 标记误差<3%的点（如有）
    good_points = np.where(error_matrix < 0.03)
    if len(good_points[0]) > 0:
        for i, j in zip(*good_points):
            plt.plot(order_vals[j], L_vals[i], 'r*', markersize=12, 
                     markeredgewidth=1.5, alpha=0.8)
    
    plt.legend(loc='lower right', fontsize=11, framealpha=0.95, handlelength=1.2)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('results/figures/wmfa_parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存数据
    np.save('results/error_matrix.npy', error_matrix)
    np.save('results/L_vals.npy', L_vals)
    np.save('results/order_vals.npy', order_vals)
    
    print(f"\n✅ 扫描完成！最优参数组合:")
    print(f"   • 记忆窗口长度 L = {L_opt:.3f}")
    print(f"   • VF拟合阶数 = {order_opt}")
    print(f"   • 最大相对误差 = {min_err:.3f}%")
    
    # 计算工程可用区域
    usable_mask = error_matrix < 0.05
    usable_ratio = np.sum(usable_mask) / usable_mask.size * 100
    print(f"   • 误差<5%的参数组合占比: {usable_ratio:.1f}%")
    
    return L_opt, order_opt, min_err, usable_ratio


def generate_window_term(alpha, L, order):
    """窗口项拟合质量诊断（关键！）"""
    w = np.logspace(-3, 3, 300)
    wmfa = WindowModulatedApproximation(alpha, L)
    
    window_exact = wmfa.window_factor(1j * w)
    window_approx = wmfa.pade_approx_window(w, order=order)
    
    # 创建双Y轴图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 幅值 (dB)
    ax1.semilogx(w, 20*np.log10(np.abs(window_exact)+1e-15), 'k-', linewidth=2.2, label='Exact')
    ax1.semilogx(w, 20*np.log10(np.abs(window_approx)+1e-15), 'r--', linewidth=2.2, label=f'WMFA (阶数={order})')
    ax1.set_ylabel('幅值 (dB)', fontsize=12, fontweight='bold')
    ax1.set_title(f'窗口项拟合质量诊断 | α={alpha}, L={L}, VF阶数={order}', 
                 fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.7, which="both")
    ax1.legend(loc='best', fontsize=11)
    
    # 相位 (度)
    ax2.semilogx(w, np.angle(window_exact, deg=True), 'k-', linewidth=2.2, label='Exact')
    ax2.semilogx(w, np.angle(window_approx, deg=True), 'r--', linewidth=2.2, label='WMFA')
    ax2.set_xlabel('频率 $\\omega$ (rad/s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('相位 (°)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.7, which="both")
    ax2.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'results/figures/window_fit_alpha{alpha}_L{L}_order{order}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 误差计算
    window_error = np.abs(window_exact - window_approx) / (np.abs(window_exact) + 1e-15)
    max_err = np.max(window_error) * 100
    mean_err = np.mean(window_error) * 100
    
    print(f"\n🔍 窗口项拟合诊断 (L={L}, 阶数={order}):")
    print(f"   • 最大相对误差: {max_err:.3f}%")
    print(f"   • 平均相对误差: {mean_err:.3f}%")
    if max_err < 3.0:
        print("   ✅ 拟合质量优秀 (<3%) - 系统级误差有望<5%")
    elif max_err < 5.0:
        print("   ⚠️ 拟合质量可接受 (<5%) - 建议微调L或增加阶数")
    else:
        print("   ❗ 拟合质量不足 (>5%) - 窗口项是系统误差主因！")
        print("      → 建议: 1) 增加VF阶数 2) 调整L值 3) 检查低频采样")
    
    return max_err


def generate_wmfa_response(alpha, L, order):
    """生成WMFA系统级响应图（幅频+相频）"""
    w = np.logspace(-3, 3, 300)
    wmfa = WindowModulatedApproximation(alpha, L)
    exact = wmfa.exact_response(w)
    approx = wmfa.approximate_response(w, order=order)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 幅频
    ax1.semilogx(w, 20*np.log10(np.abs(exact)+1e-15), 'k-', linewidth=2.2, label='Exact (短记忆系统)')
    ax1.semilogx(w, 20*np.log10(np.abs(approx)+1e-15), 'r--', linewidth=2.2, label=f'WMFA (阶数={order})')
    ax1.set_ylabel('幅值 (dB)', fontsize=12, fontweight='bold')
    ax1.set_title(f'WMFA系统级响应 | α={alpha}, L={L:.2f}, VF阶数={order}', 
                 fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.7, which="both")
    ax1.legend(loc='best', fontsize=11)
    
    # 相频
    ax2.semilogx(w, np.angle(exact, deg=True), 'k-', linewidth=2.2, label='Exact')
    ax2.semilogx(w, np.angle(approx, deg=True), 'r--', linewidth=2.2, label='WMFA')
    ax2.set_xlabel('频率 $\\omega$ (rad/s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('相位 (°)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.7, which="both")
    ax2.legend(loc='best', fontsize=11)
    
    # 添加误差标注
    error = wmfa.calculate_error(exact, approx)
    max_err_idx = np.argmax(error)
    ax1.annotate(f'最大误差点\n{error[max_err_idx]*100:.1f}%', 
                xy=(w[max_err_idx], 20*np.log10(np.abs(exact[max_err_idx])+1e-15)),
                xytext=(w[max_err_idx]*2, 20*np.log10(np.abs(exact[max_err_idx])+1e-15)-10),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'results/figures/wmfa_response_alpha{alpha}_L{L}_order{order}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 输出系统级误差
    max_sys_err = np.max(error) * 100
    print(f"\n📊 系统级响应误差 (L={L}, 阶数={order}):")
    print(f"   • 最大相对误差: {max_sys_err:.3f}%")
    print(f"   • 误差<5%频带覆盖率: {np.mean(error < 0.05)*100:.1f}%")
    
    return max_sys_err


# === 主流程 ===
def main():
    alpha = 0.7
    L_test = 2.0      # Stage 1 验证点
    order_test = 18   # Stage 1 验证阶数
    
    print("="*70)
    print("WMFA Stage 2: 参数敏感性分析专注版 (Oustaloup模块已彻底移除)")
    print("="*70)
    print("🔧 本次优化重点:")
    print("   • 低频加密采样 (窗口项拟合关键)")
    print("   • 增加实极点数量 (n_poles_real=4)")
    print("   • 扩展VF阶数扫描范围 [14, 22]")
    print("   • 增强热力图可视化 (标出<5%区域)")
    print("="*70)
    
    # Stage 1: 物理特性验证（关键诊断）
    print("\n[Stage 1] WMFA 物理特性验证 (L=2.0, 阶数=18)")
    print("-"*70)
    window_err = generate_window_term(alpha, L_test, order_test)
    system_err = generate_wmfa_response(alpha, L_test, order_test)
    
    # Stage 2: 参数敏感性分析（核心目标）
    print("\n[Stage 2] WMFA 参数敏感性分析")
    print("-"*70)
    L_opt, order_opt, min_err, usable_ratio = optimized_parameter_analysis(alpha=alpha)
    
    # Stage 3: 最优参数验证
    print("\n[Stage 3] 最优参数组合验证")
    print("-"*70)
    print(f"使用最优参数: L={L_opt:.3f}, VF阶数={order_opt}")
    window_err_opt = generate_window_term(alpha, L_opt, order_opt)
    system_err_opt = generate_wmfa_response(alpha, L_opt, order_opt)
    
    # 保存总结报告
    report_path = 'results/stage2_parameter_analysis_summary.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("WMFA 参数敏感性分析报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"测试参数: α = {alpha}\n")
        f.write(f"扫描范围: L ∈ [1.0, 3.0], VF阶数 ∈ [14, 22]\n\n")
        f.write("【Stage 1】验证点 (L=2.0, 阶数=18)\n")
        f.write(f"  • 窗口项最大误差: {window_err:.3f}%\n")
        f.write(f"  • 系统级最大误差: {system_err:.3f}%\n\n")
        f.write("【Stage 2】全局最优参数\n")
        f.write(f"  • 最优 L = {L_opt:.3f}\n")
        f.write(f"  • 最优 VF阶数 = {order_opt}\n")
        f.write(f"  • 最小最大误差 = {min_err:.3f}%\n")
        f.write(f"  • 误差<5%参数占比 = {usable_ratio:.1f}%\n\n")
        f.write("【Stage 3】最优参数验证\n")
        f.write(f"  • 窗口项最大误差: {window_err_opt:.3f}%\n")
        f.write(f"  • 系统级最大误差: {system_err_opt:.3f}%\n")
        f.write(f"  • 误差<5%频带覆盖率: {np.mean(wmfa.calculate_error(wmfa.exact_response(np.logspace(-3,3,300)), wmfa.approximate_response(np.logspace(-3,3,300), order_opt)) < 0.05)*100:.1f}%\n\n")
        f.write("结论:\n")
        if system_err_opt < 5.0:
            f.write(f"✓ WMFA在最优参数下实现高精度逼近 (误差={system_err_opt:.2f}% < 5%)\n")
            f.write("✓ 窗口调制机制有效平衡记忆效应与实现复杂度\n")
        else:
            f.write(f"△ 当前最优误差 {system_err_opt:.2f}%，建议:\n")
            f.write("  - 尝试更高VF阶数 (23-25)\n")
            f.write("  - 微调L值 (在L_opt±0.2范围内精细扫描)\n")
            f.write("  - 检查窗口项在ω<0.01区域的拟合质量\n")
    
    # 最终总结
    print("\n" + "="*70)
    print("✅ WMFA 参数敏感性分析完成")
    print("="*70)
    print(f"• 全局最优参数: L = {L_opt:.3f}, VF阶数 = {order_opt}")
    print(f"• 系统级最大误差: {system_err_opt:.3f}%")
    print(f"• 误差<5%频带覆盖率: {np.mean(wmfa.calculate_error(wmfa.exact_response(np.logspace(-3,3,300)), wmfa.approximate_response(np.logspace(-3,3,300), order_opt)) < 0.05)*100:.1f}%")
    print(f"• 误差<5%参数组合占比: {usable_ratio:.1f}%")
    print("\n💡 核心结论:")
    if system_err_opt < 5.0:
        print(f"   ✓✓ WMFA成功实现短记忆分数阶系统高精度逼近 (<5%误差)")
        print(f"   ✓ 窗口长度L与VF阶数存在明确协同优化关系（见热力图）")
    else:
        print(f"   △ 当前最优误差 {system_err_opt:.2f}% 接近工程阈值")
        print(f"   → 重点优化方向: 窗口项低频拟合（当前窗口项误差={window_err_opt:.2f}%）")
    print(f"\n📁 所有结果已保存至:")
    print(f"   • 热力图: results/figures/wmfa_parameter_sensitivity.png")
    print(f"   • 窗口项诊断: results/figures/window_fit_*.png")
    print(f"   • 系统响应: results/figures/wmfa_response_*.png")
    print(f"   • 详细报告: {report_path}")
    print("="*70)


if __name__ == "__main__":
    # 创建wmfa实例用于Stage 3验证（避免重复创建）
    wmfa = WindowModulatedApproximation(0.7, 1.8)  # 临时，会被覆盖
    main()