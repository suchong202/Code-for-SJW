"""
环境：刺入真实软组织（如猪肉）中运行，开启了模糊自适应导纳控制。
图1（轨迹退让图）：红色虚线（VPM期望目标）vs 蓝色实线（实际执行轨迹）。但这里的蓝线在受力时会主动低于红线，反映的是**“柔顺退让程度”**。
图2（接触力保护图）：反映的是传感器读到的实际受力。证明受力被完美限制在安全阈值（如3.0N）以内。
图3（自适应参数变化图）：反映刚度 K 和阻尼 B 随时间的动态变化。证明模糊逻辑在实时工作。
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# ================= 论文画图标准设置 =================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 16

COLORS = ['#E64B35', '#3C5488', '#00A087', '#4DBBD5', '#F39B7F']


def plot_admittance_results(style, trial):
    # 根据传入的 style 和 trial 去新文件夹找数据
    file_name = f"data/4_data_admittance_control_result/{style}/{trial}.csv"
    if not os.path.exists(file_name):
        print(f"❌ 找不到数据文件: {file_name}")
        return

    task_type = "Lifting" if style in [1, 2] else "Twirling"
    print(f"📊 正在绘制 Style {style} Trial {trial} 的导纳控制结果图...")
    df = pd.read_csv(file_name)
    time_s = df['Time_s'].values

    # 动态单位转换
    if task_type == "Lifting":
        vpm_target = df['VPM_Target'].values * 1000
        adm_target = df['Admittance_Target'].values * 1000
        actual = df['Actual_Pos'].values * 1000
        force = df['Force_Torque'].values
        ylabel_pos = 'Insertion Depth (mm)'
        ylabel_force = 'Force Fz (N)'
        title_pos = 'Trajectory Tracking with Compliant Retraction (Lifting)'
        title_force = 'Interaction Force Profile (Lifting)'
        # title_pos = '(A) Trajectory Tracking with Compliant Retraction (Simulated Skin)'
        # title_force = '(A) Interaction Force Profile (Simulated Skin)'
        # title_pos = '(B) Trajectory Tracking with Compliant Retraction (Real Tissue)'
        # title_force = '(B) Interaction Force Profile (Real Tissue)'
        safe_max = 3.0  # 从 Config 中抄过来的阈值
    else:
        # vpm_target = np.degrees(df['VPM_Target'].values)
        vpm_target = df['VPM_Target'].values
        adm_target = np.degrees(df['Admittance_Target'].values)
        actual = -np.degrees(df['Actual_Pos'].values)
        force = df['Force_Torque'].values
        ylabel_pos = 'Twirling Angle (°)'
        ylabel_force = 'Torque Mz (N·m)'
        title_pos = 'Trajectory Tracking with Compliant Retraction (Twirling)'
        title_force = 'Interaction Torque Profile (Twirling)'
        safe_max = 0.15

    k_val = df['Adaptive_K'].values
    b_val = df['Adaptive_B'].values

    out_dir = f"data/4_data_admittance_control_result/{style}"

    # ================= 图1：轨迹柔顺退让对比图 =================
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(time_s, vpm_target, color=COLORS[0], linestyle='--', linewidth=2.5, label='VPM Reference')
    ax1.plot(time_s, actual, color=COLORS[1], linestyle='-', linewidth=2, label='Actual Compliant Execution')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(ylabel_pos)
    ax1.set_title(title_pos, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    fig1.savefig(f"{out_dir}/Fig_Admittance_Trajectory_S{style}_T{trial}.png", dpi=300)
    fig1.savefig(f"{out_dir}/Fig_Admittance_Trajectory_S{style}_T{trial}.pdf")
    # fig1.savefig(f"plot/4_1_Fig_Admittance_Trajectory_{task_type}.png", dpi=300)
    # fig1.savefig(f"plot/4_1_Fig_Admittance_Trajectory_{task_type}.pdf")

    # ================= 图2：接触力/力矩保护曲线 =================
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(time_s, force, color=COLORS[2], linewidth=2, label=ylabel_force)
    ax2.axhline(y=safe_max, color='gray', linestyle='-.', linewidth=2, label='Safety Threshold')
    ax2.axhline(y=-safe_max, color='gray', linestyle='-.', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(ylabel_force)
    ax2.set_title(title_force, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    fig2.savefig(f"{out_dir}/Fig_Admittance_Force_S{style}_T{trial}.png", dpi=300)
    fig2.savefig(f"{out_dir}/Fig_Admittance_Force_S{style}_T{trial}.pdf")
    # fig2.savefig(f"plot/4_2_Fig_Admittance_Force_{task_type}.png", dpi=300)
    # fig2.savefig(f"plot/4_2_Fig_Admittance_Force_{task_type}.pdf")

    # ================= 图3：自适应参数变化图 =================
    fig3, ax3a = plt.subplots(figsize=(10, 5))
    color_k, color_b = COLORS[3], COLORS[4]

    ax3a.set_xlabel('Time (s)')
    ax3a.set_ylabel('Adaptive Stiffness K', color=color_k)
    line1 = ax3a.plot(time_s, k_val, color=color_k, linewidth=2, label='Stiffness (K)')
    ax3a.tick_params(axis='y', labelcolor=color_k)

    ax3b = ax3a.twinx()
    ax3b.set_ylabel('Adaptive Damping B', color=color_b)
    line2 = ax3b.plot(time_s, b_val, color=color_b, linewidth=2, linestyle='--', label='Damping (B)')
    ax3b.tick_params(axis='y', labelcolor=color_b)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3b.legend(lines, labels, loc='upper right', framealpha=0.7)

    ax3a.set_title(f'Online Adaptation of Impedance Parameters ({task_type})', fontweight='bold')
    # ax3a.set_title(f'(A) Online Adaptation of Impedance Parameters (Simulated Skin)', fontweight='bold')
    # ax3a.set_title(f'(B) Online Adaptation of Impedance Parameters (Real Tissue)', fontweight='bold')
    ax3a.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    fig3.savefig(f"{out_dir}/Fig_Admittance_Params_S{style}_T{trial}.png", dpi=300)
    fig3.savefig(f"{out_dir}/Fig_Admittance_Params_S{style}_T{trial}.pdf")
    # fig3.savefig(f"plot/4_3_Fig_Admittance_Parameters_{task_type}.png", dpi=300)
    # fig3.savefig(f"plot/4_3_Fig_Admittance_Parameters_{task_type}.pdf")

    print(f"✅ {task_type} 相关的 3 张图表已保存！\n")
    plt.show()


if __name__ == "__main__":
    # 指定你想看哪个手法、哪一次操作的导纳结果
    # 比如查看 Style 1 (提插补) 的第 1 次操作
    plot_admittance_results(style=1, trial=10)