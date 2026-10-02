import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# ================= SCI 期刊画图标准设置 =================
# 字体设为 Times New Roman，字号设为 12
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.serif'] = ['Calibri']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# 设定学术风高级配色 (科技蓝、砖红、翠绿、浅紫)
COLORS = ['#3C5488', '#E64B35', '#00A087', '#4DBBD5']


def plot_trajectory():
    """图1：30s 连续轨迹预测对比图 (LSTM vs MLP vs Ground Truth)"""
    print("正在生成轨迹对比图...")
    try:
        # 分别读取提插 (Lifting) 和捻转 (Twirling) 的数据
        gt_z = np.load("gt_Lifting.npy")
        pred_lstm_z = np.load("LSTM_Lifting_preds.npy")
        pred_mlp_z = np.load("MLP_Lifting_preds.npy")

        gt_theta = np.load("gt_Twirling.npy")
        pred_lstm_theta = np.load("LSTM_Twirling_preds.npy")
        pred_mlp_theta = np.load("MLP_Twirling_preds.npy")
    except FileNotFoundError:
        print("⚠️ 未找到分离的 .npy 轨迹文件，请确保主程序已运行并保存了双流数据！")
        return

    # 生成时间轴 (10Hz，共约 30s)
    time_axis_z = np.arange(len(gt_z)) * 0.1
    time_axis_theta = np.arange(len(gt_theta)) * 0.1

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 子图 1: 提插深度 (Z轴)
    axes[0].plot(time_axis_z, gt_z[:, 0] * 1000, color='black', linestyle='--', linewidth=2,
                 label='Ground Truth')  # 乘以1000转换为毫米
    axes[0].plot(time_axis_z, pred_mlp_z[:, 0] * 1000, color=COLORS[3], alpha=0.8, linewidth=1.5, label='MLP Baseline')
    axes[0].plot(time_axis_z, pred_lstm_z[:, 0] * 1000, color=COLORS[1], alpha=0.9, linewidth=2, label='LSTM (Proposed)')
    axes[0].set_ylabel('LT Depth $\Delta Z$ (mm)')
    axes[0].set_title('(A) LT MAM Trajectory Prediction', loc='left', fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.7)
    axes[0].legend(loc='upper right')

    # 子图 2: 捻转角度 (Theta)
    # 注意：如果捻转全是噪声，展示这一张图刚好可以配合论文里的 limitation 讨论
    axes[1].plot(time_axis_theta, gt_theta[:, 0], color='black', linestyle='--', linewidth=2,
                 label='Ground Truth')  # 转换为角度
    axes[1].plot(time_axis_theta, pred_mlp_theta[:, 0], color=COLORS[3], alpha=0.8, linewidth=1.5, label='MLP Baseline')
    axes[1].plot(time_axis_theta, pred_lstm_theta[:, 0], color=COLORS[1], alpha=0.9, linewidth=2,
                 label='LSTM (Proposed)')
    axes[1].set_ylabel('TR Angle $\Delta \\theta$ (°)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('(B) TR MAM Trajectory Prediction', loc='left', fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig("Fig_Trajectory_Comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig("Fig_Trajectory_Comparison.pdf", bbox_inches='tight')
    plt.close()


def plot_boxplot():
    """图2：四种针刺手法的 RMSE 箱线图 (验证泛化能力)"""
    print("正在生成 RMSE 箱线图...")
    df = pd.read_csv("experiment_results.csv")

    # 只抽取 LSTM 模型的结果
    df_lstm = df[df['Model'] == 'LSTM'].copy()

    # 映射英文标签
    style_map = {
        1: 'RFLT',
        2: 'RDLT',
        3: 'RFTR',
        4: 'RDTR'
    }
    df_lstm['Manipulation'] = df_lstm['Style'].map(style_map)
    df_lstm['RMSE_Z_mm'] = df_lstm['RMSE_Z'] * 1000  # 换算为毫米
    df_lstm['RMSE_Theta_deg'] = np.degrees(df_lstm['RMSE_Theta'])  # 换算为角度

    # 把数据拆成提插和捻转两个 DataFrame，自动过滤掉 NaN
    df_lift = df_lstm[df_lstm['Task'] == 'Lifting']
    df_twirl = df_lstm[df_lstm['Task'] == 'Twirling']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # 子图 1: 提插 RMSE
    sns.boxplot(data=df_lift, x='Manipulation', y='RMSE_Z_mm', ax=axes[0],
                hue='Manipulation', legend=False, palette="Blues", width=0.5, boxprops=dict(alpha=0.8))
    sns.stripplot(data=df_lift, x='Manipulation', y='RMSE_Z_mm', ax=axes[0],
                  color=".2", size=5, alpha=0.6)  # 添加数据散点
    axes[0].set_ylabel('RMSE of LT Depth (mm)')
    axes[0].set_xlabel('')
    axes[0].set_title('(A) Error Distribution in LT MAM', loc='left', fontweight='bold')

    # 子图 2: 捻转 RMSE
    sns.boxplot(data=df_twirl, x='Manipulation', y='RMSE_Theta_deg', ax=axes[1],
                hue='Manipulation', legend=False, palette="Reds", width=0.5, boxprops=dict(alpha=0.8))
    sns.stripplot(data=df_twirl, x='Manipulation', y='RMSE_Theta_deg', ax=axes[1],
                  color=".2", size=5, alpha=0.6)
    axes[1].set_ylabel('RMSE of TR Angle (°)')
    axes[1].set_xlabel('')
    axes[1].set_title('(B) Error Distribution in TR MAM', loc='left', fontweight='bold')

    plt.tight_layout()
    plt.savefig("Fig_RMSE_Boxplot.png", dpi=300, bbox_inches='tight')
    plt.savefig("Fig_RMSE_Boxplot.pdf", bbox_inches='tight')
    plt.close()


def plot_ablation_barchart():
    """图3：特征消融实验的 R 值柱状图"""
    print("正在生成消融实验柱状图...")
    df_ab = pd.read_csv("ablation_results.csv")

    # 重命名标签使其更正式
    mode_map = {
        'skeleton_only': 'Model A\nSkeleton feature only',
        'skeleton_wrist': 'Model B\n+ Macroscopic anchors',
        'full': 'Model C\nFull model(Proposed)'
    }
    df_ab['Feature_Mode'] = df_ab['Feature_Mode'].map(mode_map)

    # 按照特定顺序排列
    order = ['Model A\nSkeleton feature only', 'Model B\n+ Macroscopic anchors', 'Model C\nFull model(Proposed)']

    # 计算均值和标准差
    # grouped = df_ab.groupby('Feature_Mode')[['R_Z', 'R_Theta']].agg(['mean', 'std']).reindex(order)
    grouped = df_ab.groupby('Feature_Mode')[['R_Z', 'R_Theta']].mean(numeric_only=True).reindex(order)
    grouped_std = df_ab.groupby('Feature_Mode')[['R_Z', 'R_Theta']].std(numeric_only=True).reindex(order)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(order))
    width = 0.35

    # 画柱状图并带有误差棒
    bar1 = ax.bar(x - width / 2, grouped['R_Z'], width,
                  yerr=grouped_std['R_Z'], capsize=5,
                  label='LT MAM (Z-axis)', color=COLORS[0], alpha=0.85, edgecolor='black')

    bar2 = ax.bar(x + width / 2, grouped['R_Theta'], width,
                  yerr=grouped_std['R_Theta'], capsize=5,
                  label='TR MAM (Theta)', color=COLORS[1], alpha=0.85, edgecolor='black')

    ax.set_ylabel('Pearson Correlation Coefficient ($R$)')
    ax.set_title('Ablation Study on Explicit Prior Features', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle=':', alpha=0.7)

    # 设置 y 轴范围 (0 ~ 1)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig("Fig_Ablation_Barchart.png", dpi=300, bbox_inches='tight')
    plt.savefig("Fig_Ablation_Barchart.pdf", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("🚀 开始绘制 SCI 论文级图表...")
    plot_trajectory()
    plot_boxplot()
    plot_ablation_barchart()
    print("✅ 绘图完成！请查看当前目录下的 .png 和 .pdf 文件。")