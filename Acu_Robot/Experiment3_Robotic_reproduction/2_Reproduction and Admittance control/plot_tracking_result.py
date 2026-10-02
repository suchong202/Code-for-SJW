"""
改动：
新增了一个评估函数：把文件夹里 _1.csv 到 _5.csv 的 5 个文件全读一遍，算出一个 Mean ± SD (均值±标准差) 的误差输出给你写论文用。
自动挑最均值的曲线画图：算出平均误差后，挑一个误差最接近均值的 Trial，把它的编号传给画图函数去画图
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
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12



def evaluate_tracking_error(target, actual, task_type):
    """计算轨迹跟踪的量化误差指标"""
    # 计算误差绝对值序列
    error_seq = np.abs(target - actual)

    # 计算 RMSE 和 Max Error
    rmse = np.sqrt(np.mean(error_seq ** 2))
    max_error = np.max(error_seq)

    unit = "mm" if task_type == "Lifting" else "°"
    print(f"================ {task_type} 轨迹跟踪误差评估 ================")
    print(f"👉 跟踪均方根误差 (Tracking RMSE): {rmse:.4f} {unit}")
    print(f"👉 最大绝对跟踪误差 (Max Tracking Error): {max_error:.4f} {unit}")
    print("==============================================================\n")


def evaluate_tracking_multiple_trials(style, num_trials=10):
    """读取多个Trial，计算平均误差，并返回最接近均值的代表性Trial序号"""
    task_type = "Lifting" if style in [1, 2] else "Twirling"
    rmse_list, max_err_list, valid_trials = [], [], []

    # 1. 遍历读取所有 Trial 数据
    for t in range(1, num_trials + 1):
        # 按照新结构去对应文件夹读取
        file_name = f"data/3_data_tracking_result/{style}/{t}.csv"
        if not os.path.exists(file_name): continue

        df = pd.read_csv(file_name)

        if task_type == "Lifting":
            target = df['Target_Z_m'].values * 1000
            actual = df['Actual_Z_m'].values * 1000
        else:
            target = df['Target_Theta_rad'].values              # target 本身是角度，直接读
            actual = -np.degrees(df['Actual_Theta_rad'].values) # actual 是弧度且反向，转角度并翻转

        error_seq = np.abs(target - actual)
        rmse = np.sqrt(np.mean(error_seq ** 2))
        max_error = np.max(error_seq)

        rmse_list.append(rmse)
        max_err_list.append(max_error)
        valid_trials.append(t)

    if not valid_trials:
        print(f"❌ 找不到任何 {task_type} 的轨迹文件，请检查命名格式！(如 robot_tracking_result_Lifting_1.csv)")
        return None

    # 2. 计算统计学均值与标准差 (Mean ± SD)
    mean_rmse, std_rmse = np.mean(rmse_list), np.std(rmse_list)
    mean_max, std_max = np.mean(max_err_list), np.std(max_err_list)

    unit = "mm" if task_type == "Lifting" else "°"
    print(f"================ {task_type} 轨迹跟踪综合误差评估 ({len(valid_trials)}次试验) ================")
    print(f"👉 跟踪均方根误差 (RMSE): {mean_rmse:.4f} ± {std_rmse:.4f} {unit}")
    print(f"👉 最大跟踪误差 (Max Error): {mean_max:.4f} ± {std_max:.4f} {unit}")

    # 3. 寻找最接近平均 RMSE 的一次 Trial 作为代表去画图
    diff_from_mean = np.abs(np.array(rmse_list) - mean_rmse)
    best_idx = np.argmin(diff_from_mean)
    rep_trial = valid_trials[best_idx]

    print(f"👉 自动选取最接近均值的代表性 Trial: 第 {rep_trial} 次")
    print("========================================================================\n")

    return rep_trial


def plot_tracking(style, rep_trial):
    """绘制代表性 Trial 的轨迹图"""
    if rep_trial is None: return
    task_type = "Lifting" if style in [1, 2] else "Twirling"

    # 按照新结构去读取
    file_name = f"data/3_data_tracking_result/{style}/{rep_trial}.csv"
    print(f"📊 正在绘制 {task_type} 轨迹跟踪对比图并计算误差...")
    df = pd.read_csv(file_name)
    time_s = df['Time_s'].values

    # ================= 图 A 时间轨迹图 =================
    fig1, ax1 = plt.subplots(figsize=(10, 4))

    if task_type == "Lifting":
        # 提插任务：米 转换为 毫米
        target = df['Target_Z_m'].values * 1000
        actual = df['Actual_Z_m'].values * 1000

        # 打印量化误差指标
        evaluate_tracking_error(target, actual, task_type)

        # 画图：红色虚线为 Target，蓝色实线为 Actual
        ax1.plot(time_s, target, color='#E64B35', linestyle='--', linewidth=2.5, label='Target Trajectory (VPM)')
        ax1.plot(time_s, actual, color='#3C5488', linestyle='-', linewidth=2, label='Actual Execution (Robot)')
        ax1.set_ylabel('Lifting Depth (mm)')
        ax1.set_title('(A) Kinematic Tracking Performance in LT MAM', fontweight='bold', pad=15, loc='left')

    elif task_type == "Twirling":
        # 捻转任务：弧度 转换为 角度
        # target = np.degrees(df['Target_Theta_rad'].values)
        actual = -np.degrees(df['Actual_Theta_rad'].values)
        target = df['Target_Theta_rad'].values
        # actual = df['Actual_Theta_rad'].values

        # 打印量化误差指标
        evaluate_tracking_error(target, actual, task_type)

        # 画图：红色虚线为 Target，蓝色实线为 Actual
        ax1.plot(time_s, target, color='#E64B35', linestyle='--', linewidth=2.5, label='Target Trajectory (VPM)')
        ax1.plot(time_s, actual, color='#3C5488', linestyle='-', linewidth=2, label='Actual Execution (Robot)')
        ax1.set_ylabel('Rotating Angle (°)')
        ax1.set_title('(B) Kinematic Tracking Performance in TR MAM', fontweight='bold', pad=15, loc='left')

    ax1.set_xlabel('Time (s)')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.tick_params(direction='in', length=5, width=1)


    # ================= 图 B 一致性散点图 =================
    fig2, ax2 = plt.subplots(figsize=(6, 6))

    # 理想对角线
    min_val = min(np.min(target), np.min(actual))
    max_val = max(np.max(target), np.max(actual))
    margin = (max_val - min_val) * 0.1
    ax2.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
             color='gray', linestyle='--', linewidth=1.5, label='Ideal Tracking (y=x)')

    # 散点
    ax2.scatter(target, actual, color='#00A087', alpha=0.6, s=20, edgecolors='none', label='Sampled Poses')

    # 计算 R^2
    r_matrix = np.corrcoef(target, actual)
    r_squared = r_matrix[0, 1] ** 2

    # 按照要求设置标题
    title_B = f"(A) Master-Slave Consistency in LT MAM\n($R^2={r_squared:.4f}$)" if task_type == 'Lifting' else f"(B) Master-Slave Consistency in TR MAM\n($R^2={r_squared:.4f}$)"
    ax2.set_title(title_B, fontweight='bold', pad=15)

    unit = "mm" if task_type == 'Lifting' else "°"
    ax2.set_xlabel(f'Target Trajectory ({unit})')
    ax2.set_ylabel(f'Actual Execution ({unit})')

    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.tick_params(direction='in', length=5, width=1)

    # ================= 分别保存两张高清图表 =================
    out_dir = f"data/3_data_tracking_result/{style}"
    os.makedirs(out_dir, exist_ok=True)  # 确保文件夹存在

    fig1.tight_layout()
    fig1.savefig(f"{out_dir}/Overlay_Style{style}_Trial{rep_trial}.png", dpi=300, bbox_inches='tight')
    fig1.savefig(f"{out_dir}/Overlay_Style{style}_Trial{rep_trial}.pdf", bbox_inches='tight')

    fig2.tight_layout()
    fig2.savefig(f"{out_dir}/Scatter_Style{style}_Trial{rep_trial}.png", dpi=300, bbox_inches='tight')
    fig2.savefig(f"{out_dir}/Scatter_Style{style}_Trial{rep_trial}.pdf", bbox_inches='tight')

    print(f"✅ Overlay 图与 Scatter 图已保存至: {out_dir}")
    plt.show()



if __name__ == "__main__":
    # ================= 模式 1：自动评估并画出最代表性的图 =================
    print("【模式 1：自动统计与绘图】")
    for style_to_plot in [1, 3]:
        # 计算 10 次的平均误差，并返回最接近均值的 trial 序号
        rep_trial = evaluate_tracking_multiple_trials(style=style_to_plot, num_trials=10)
        # 画图
        plot_tracking(style=style_to_plot, rep_trial=rep_trial)

    # ================= 模式 2：手动指定画哪一张图 =================
    print("\n【模式 2：手动指定绘图】")

    # 假设你看了数据，觉得 Style 1 的 Trial 5 画出来最漂亮，想单独画它：
    # 直接调用 plot_tracking 即可
    MANUAL_STYLE = 1
    MANUAL_TRIAL = 1

    print(f"正在手动绘制 Style {MANUAL_STYLE}, Trial {MANUAL_TRIAL} ...")
    plot_tracking(style=MANUAL_STYLE, rep_trial=MANUAL_TRIAL)

    # 同理，你可以单独画捻转的某个图：
    # plot_tracking(style=3, rep_trial=2)