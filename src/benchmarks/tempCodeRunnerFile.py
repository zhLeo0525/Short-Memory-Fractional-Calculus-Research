    # =============== 生成图表 ===============
    print("生成图表...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1：时域响应对比
    ax1 = axes[0, 0]
    ax1.plot(t, analytical_inf, 'k--', linewidth=2, label='解析解 (无限记忆)')
    ax1.plot(t, results['infinite_numerical'], 'b-', linewidth=1.5, label='数值解 (无限记忆)')
    colors = ['r', 'g', 'm', 'c']
    for i, L in enumerate(L_values):
        ax1.plot(t, results['short_memory'][L], colors[i], 
                label=f'短记忆 (L={L})', alpha=0.8)
    ax1.set_xlabel('时间 t')
    ax1.set_ylabel('积分输出')
    ax1.set_title(f'阶跃响应对比 (α={alpha})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 图2：短记忆与无限记忆差异（放大关键区域）
    ax2 = axes[0, 1]
    for i, L in enumerate(L_values[:-1]):  # 排除L=5.0
        diff = results['short_memory'][L] - results['infinite_numerical']
        ax2.plot(t, diff, colors[i], label=f'L={L}', alpha=0.8)
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('时间 t')
    ax2.set_ylabel('输出差异')
    ax2.set_title('短记忆与无限记忆的差异')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 3)  # 聚焦初始阶段
    
    # 图3：相对误差随时间变化
    ax3 = axes[1, 0]
    for i, L in enumerate(L_values[:-1]):
        ax3.plot(t, errors[L], colors[i], label=f'L={L}', alpha=0.8)
    ax3.set_xlabel('时间 t')
    ax3.set_ylabel('相对误差')
    ax3.set_yscale('log')  # 对数坐标更清晰
    ax3.set_title('相对误差 (对数坐标)')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend()
    
    # 图4：误差与L的关系（在固定时间点）
    ax4 = axes[1, 1]
    t_eval = [1.0, 2.0, 3.0, 4.0]  # 评估时间点
    for te in t_eval:
        idx = np.argmin(np.abs(t - te))
        err_at_t = [errors[L][idx] for L in L_values[:-1]]
        ax4.plot(L_values[:-1], err_at_t, 'o-', label=f't={te}')
    ax4.set_xlabel('记忆窗口长度 L')
    ax4.set_ylabel('相对误差')
    ax4.set_yscale('log')
    ax4.set_title(f'误差随L的变化 (α={alpha})')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend()