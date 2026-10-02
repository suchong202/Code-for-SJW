import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import os

# ---------------- 配置参数 ----------------
# 设置中文字体以防备用，但本代码图表已全英文输出
# plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'Heiti TC']
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR_1 = 'data_pos_1'  # without force feedback
DATA_DIR_2 = 'data_pos_2'  # with force feedback
SAMPLING_RATE = 10  # 10Hz
DT = 1 / SAMPLING_RATE

# ================= 核心切换开关 =================
# 可选值: 'Lifting' (提插) 或 'Twisting' (捻转)
ANALYSIS_TYPE = 'Lifting'
# ANALYSIS_TYPE = 'Twisting'
# ================================================

# 自动根据手法类型配置参数
if ANALYSIS_TYPE == 'Lifting':
    TECHNIQUES = [(1, 'RFLT'), (2, 'RDLT')]
    MAIN_TITLE_PREFIX = 'LT MAM'
    POS_COL = 'Robot_Z'
    FORCE_COL = 'Fz'
    POS_LABEL = 'Z-axis Position (m)'
    FORCE_LABEL = 'Axial Force Fz (N)'
    VELOCITY_LABEL = 'Absolute Velocity (m/s)'
    PHASE_1_NAME, PHASE_2_NAME = 'Insert Phase', 'Withdraw Phase'
else:
    TECHNIQUES = [(3, 'RFTR'), (4, 'RDTR')]
    MAIN_TITLE_PREFIX = 'TR MAM'
    POS_COL = 'Robot_Joint6'
    FORCE_COL = 'Mz'
    POS_LABEL = 'Rotation Angle (°)'
    FORCE_LABEL = 'Torque Mz (Nm)'
    VELOCITY_LABEL = 'Absolute Angular Velocity (°/s)'
    PHASE_1_NAME, PHASE_2_NAME = 'Forward Rotation', 'Backward Rotation'


def load_data(base_dir, technique_idx, trial_idx):
    """读取并统一5列CSV文件格式"""
    file_path = os.path.join(base_dir, str(technique_idx), f"{trial_idx}.csv")
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    # 统一命名5列格式
    cols = ['Index', 'Robot_Z', 'Robot_Joint6', 'Fz', 'Mz']
    df.columns = cols[:len(df.columns)]
    return df


# ---------------- 1. 稳定性对比分析 ----------------
def analyze_stability():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    tech_idx = TECHNIQUES[0][0]  # 选取第一种手法(补)作为对比示例

    for i, (dir_path, label) in enumerate([(DATA_DIR_1, '(A) No Force Feedback'),
                                           (DATA_DIR_2, '(B) Force Feedback')]):
        df = load_data(dir_path, tech_idx, 7)  # 取第7次操作
        if df is None: continue

        time = np.arange(len(df)) * DT
        # 减去第0个点的数据，将绝对坐标转化为相对坐标
        raw_signal = df[POS_COL]
        signal = raw_signal - raw_signal.iloc[0]
        # 计算趋势线（基线漂移）
        mean_trend = signal.rolling(window=50, center=True).mean()

        axes[i].plot(time, signal, label='Original Trajectory', color='blue', alpha=0.6)
        axes[i].plot(time, mean_trend, label='Mean Trend (Drift)', color='red', linewidth=2)

        axes[i].set_title(f'{label}', fontsize=20, fontweight='bold')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel(POS_LABEL)
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('Fig1_Stability_Comparison.png', dpi=300)
    plt.show()


# ---------------- 2. 手法辨识度分析 ----------------
def analyze_technique_kinematics():
    results = []

    for tech_idx, tech_name in TECHNIQUES:
        for trial_idx in range(1, 6):
            df = load_data(DATA_DIR_2, tech_idx, trial_idx)
            if df is None: continue

            # 计算速度 (v = dx/dt)
            v = np.diff(df[POS_COL]) / DT

            # 区分正负相位（插提 / 正反转）
            v_phase1 = v[v < 0] if ANALYSIS_TYPE == 'Lifting' else v[v > 0]
            v_phase2 = v[v > 0] if ANALYSIS_TYPE == 'Lifting' else v[v < 0]

            results.append({
                'Technique': tech_name,
                'Trial': trial_idx,
                PHASE_1_NAME: np.abs(np.mean(v_phase1)) if len(v_phase1) > 0 else 0,
                PHASE_2_NAME: np.abs(np.mean(v_phase2)) if len(v_phase2) > 0 else 0
            })

    res_df = pd.DataFrame(results)
    if res_df.empty: return

    # 绘制速度对比柱状图 (Seaborn风格)
    melted_df = res_df.melt(id_vars='Technique', value_vars=[PHASE_1_NAME, PHASE_2_NAME],
                            var_name='Phase', value_name='Velocity')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_df, x='Technique', y='Velocity', hue='Phase', capsize=.1)

    plt.title(f'(A) Statistics of Kinematic Asymmetry for {MAIN_TITLE_PREFIX}', fontsize=20, fontweight='bold')
    # plt.title(f'(B) Statistics of Kinematic Asymmetry for {MAIN_TITLE_PREFIX}', fontsize=20, fontweight='bold')
    plt.ylabel(VELOCITY_LABEL)
    plt.xlabel('')
    # plt.xlabel('Manipulation Technique')
    plt.legend(title='')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('Fig2_Velocity_Asymmetry.png', dpi=300)
    plt.show()


# ---------------- 3. 力-位移迟滞环 ----------------
def plot_hysteresis_loop():
    plt.figure(figsize=(12, 5))

    for idx, (tech_idx, tech_name) in enumerate(TECHNIQUES):
        plt.subplot(1, 2, idx + 1)
        for trial_idx in range(1, 6):
            df = load_data(DATA_DIR_2, tech_idx, trial_idx)
            if df is None: continue

            # X: 深度/角度, Y: 力/力矩
            x_smooth = savgol_filter(df[POS_COL], 7, 2)
            y_smooth = savgol_filter(df[FORCE_COL], 7, 2)

            # 保留代码1的美术风格：alpha=0.5, linewidth=1, 不指定颜色自动循环
            plt.plot(x_smooth, y_smooth, alpha=0.5, linewidth=1)

        plt.title(f"Hysteresis Loops of {tech_name} MAM", fontsize=18, fontweight='bold')
        plt.xlabel(POS_LABEL)
        plt.ylabel(FORCE_LABEL)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Fig3_Hysteresis_Loops.png', dpi=300)
    plt.show()


# ---------------- 执行分析 ----------------
if __name__ == "__main__":
    if os.path.exists(DATA_DIR_1) and os.path.exists(DATA_DIR_2):
        print(f"当前分析模式: 【{ANALYSIS_TYPE}】")

        print("1/3 正在生成：稳定性对比分析图...")
        analyze_stability()

        print("2/3 正在生成：速度不对称性分析图...")
        analyze_technique_kinematics()

        print("3/3 正在生成：力-位移迟滞环...")
        plot_hysteresis_loop()

        print("所有图表已按设定格式生成并保存至当前目录。")
    else:
        print("错误：未找到数据文件夹 data1 或 data2，请检查路径结构。")
