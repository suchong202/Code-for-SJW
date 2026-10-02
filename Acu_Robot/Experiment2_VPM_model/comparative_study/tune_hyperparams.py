import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 定义超参数搜索空间 =================
SEARCH_SPACE = {
    'W': [5, 10, 15],  # 窗口大小: 0.5秒, 1秒, 1.5秒
    'LR': [0.001, 0.005, 0.01],  # 学习率
    'HIDDEN_DIM': [64, 128],  # 隐藏层神经元个数
    'EPOCHS': [50, 100, 150]
}

class Config:
    USE_PRIOR_FEATURES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EPOCHS = 100  # 调优时可适当减少Epoch以节省时间
BATCH_SIZE = 32


# (此处将之前的 smooth_data, get_dynamic_shift, load_and_preprocess_trial, create_windows 复制过来)
# === 保持与主程序完全一致的数据预处理函数 ===
def smooth_data(data, span=5):
    """指数移动平均(EMA)进行因果滤波"""
    df = pd.DataFrame(data)
    # EMA 仅依赖当前及历史数据，满足遥操作实时性要求
    ema_data = df.ewm(span=span, adjust=False).mean().values
    return ema_data


def get_dynamic_shift(v_vision, v_robot, max_shift=10):
    """
    带去极值鲁棒处理的动态对齐，通过互相关分析，动态计算当前 Trial 视觉与机械臂的最佳滞后帧数。
    max_shift: 允许的最大搜索帧数（防止因为纯噪声对齐到几十帧以外去）
    """
    if np.std(v_vision) == 0 or np.std(v_robot) == 0:
        return 0

        # 局部去极值与标准化（仅用于算对齐，不改变喂给LSTM的真实数据）

    def robust_normalize(v):
        p1, p99 = np.percentile(v, 1), np.percentile(v, 99)
        v_clipped = np.clip(v, p1, p99)
        if np.std(v_clipped) == 0: return v_clipped
        return (v_clipped - np.mean(v_clipped)) / np.std(v_clipped)

    v_v_norm = robust_normalize(v_vision)
    v_r_norm = robust_normalize(v_robot)

    correlation = signal.correlate(v_r_norm, v_v_norm, mode='full')
    lags = signal.correlation_lags(len(v_r_norm), len(v_v_norm), mode='full')

    valid_idx = np.where((lags >= -max_shift) & (lags <= max_shift))[0]
    best_idx = valid_idx[np.argmax(np.abs(correlation[valid_idx]))]

    return lags[best_idx]


def load_and_preprocess_trial(hand_path, pos_path):
    """读取单次 Trial 数据并进行特征工程"""
    # 1. 读取视觉数据
    df_hand = pd.read_csv(hand_path)
    # 提取第2-28列 (27维)
    hand_raw = df_hand.iloc[:, 1:28].values
    hand_raw = smooth_data(hand_raw, span=5)     # 滤波处理
    N = hand_raw.shape[0]
    # 重塑为 (N, 9, 3) 方便按节点操作
    nodes = hand_raw.reshape(N, 9, 3)
    # 提取手腕绝对位移作为关键特征（不仅仅用相对坐标）
    wrist_y = -nodes[:, 0, 1:2]  # 获取手腕Y轴绝对坐标
    node_0 = nodes[:, 0, :]  # 手腕
    node_4 = nodes[:, 4, :]  # 拇指指尖
    node_8 = nodes[:, 8, :]  # 食指指尖

    # --- 特征提取 ---
    # a. 局部坐标归一化 (减去 Node 0)
    rel_nodes = nodes[:, 1:, :] - nodes[:, 0:1, :]  # (N, 8, 3)
    rel_features = rel_nodes.reshape(N, 24)  # 展开为 24 维

    # b. 先验特征提取
    # 捏力 d_pinch (Node 8 和 Node 4 的欧氏距离)
    d_pinch = np.linalg.norm(node_8 - node_4, axis=1, keepdims=True)

    # 提插速度 v_y (Node 0 的 Y轴 一阶差分)
    # 假设 Y 轴是坐标的第二个分量 (index 1)
    v_y = np.zeros((N, 1))
    v_y[1:, 0] = -(nodes[1:, 0, 1] - nodes[:-1, 0, 1])

    # 捻转角速度 v_thumb_x (大拇指 X 轴的速度)
    rel_thumb = node_4 - node_0
    v_thumb_x = np.zeros((N, 1))
    v_thumb_x[1:, 0] = -(rel_thumb[1:, 0] - rel_thumb[:-1, 0]) # (加负号看波形情况)
    v_thumb_x = smooth_data(v_thumb_x, span=10)

    if Config.USE_PRIOR_FEATURES:
        X_features = np.concatenate([rel_features, wrist_y, d_pinch, v_y, v_thumb_x], axis=1)  # 28维
    else:
        X_features = rel_features  # 24 维 (消融实验：仅骨架)

    # 2. 读取位姿数据
    df_pos = pd.read_csv(pos_path)
    pos_raw = df_pos.iloc[:, 1:3].values # [Z, theta]

    # 转换为增量 (差分)
    Y_diff = np.zeros((N, 2))
    Y_diff[1:, :] = pos_raw[1:, :] - pos_raw[:-1, :]

    # 3.时间对齐
    # 提插滞后改为动态获取 (由于提插动作规律性强，搜索范围限制在 ±5 帧即可)
    lag_z = get_dynamic_shift(v_y.flatten(), Y_diff[:, 0].flatten(), max_shift=5)

    # 捻转滞后通过函数动态获取 (限制在 ±10 帧以内防跑飞)
    lag_theta = get_dynamic_shift(v_thumb_x.flatten(), Y_diff[:, 1].flatten(), max_shift=10)

    # 以视觉特征 X 为统一时间锚点，计算安全的切片范围
    t_start = max(0, -lag_z, -lag_theta)

    N_frames = len(X_features)
    # 计算最大有效长度，确保 X 和 Y 切片都不会越界
    length = min(N_frames - t_start,
                 N_frames - (t_start + lag_z),
                 N_frames - (t_start + lag_theta))

    # 分别进行切片重组
    X_aligned = X_features[t_start: t_start + length]
    Y_aligned = np.zeros((length, 2))

    # 提插标签按 lag_z 偏移
    Y_aligned[:, 0] = Y_diff[t_start + lag_z: t_start + length + lag_z, 0]
    # 捻转标签按独立的 lag_theta 偏移
    Y_aligned[:, 1] = Y_diff[t_start + lag_theta: t_start + length + lag_theta, 1]

    return X_aligned, Y_aligned


def create_windows(X, Y, W):
    """构建时间滑动窗口"""
    X_win, Y_win = [], []
    for i in range(len(X) - W + 1):
        X_win.append(X[i:i + W, :])
        Y_win.append(Y[i + W - 1, :])  # Y取窗口最后一帧的对应标签
    return np.array(X_win), np.array(Y_win)


# ================= 2. 支持动态调整的 LSTM 模型 =================
class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_and_eval(W, LR, HIDDEN_DIM, EPOCHS):
    """在一组特定超参数下，使用固定验证集评估模型"""
    # 我们抽取代表性的 Style 1 和 Style 3
    styles = [1, 3]
    train_trials = [1, 2, 3]  # 训练集
    val_trial = 4  # 验证集 (完全没有见过)

    train_X_raw, train_Y_raw = [], []
    val_X_raw, val_Y_raw = [], []

    for style in styles:
        for t in train_trials + [val_trial]:
            h_path, p_path = f"data/data_hand/{style}/{t}.csv", f"data/data_pos/{style}/{t}.csv"
            if not os.path.exists(h_path): continue

            X, Y = load_and_preprocess_trial(h_path, p_path)
            if t == val_trial:
                val_X_raw.append(X);
                val_Y_raw.append(Y)
            else:
                train_X_raw.append(X);
                train_Y_raw.append(Y)

    # 归一化
    scaler_X, scaler_Y = StandardScaler(), StandardScaler()
    train_X_scaled = scaler_X.fit_transform(np.vstack(train_X_raw))
    train_Y_scaled = scaler_Y.fit_transform(np.vstack(train_Y_raw))
    val_X_scaled = scaler_X.transform(np.vstack(val_X_raw))
    val_Y_scaled = scaler_Y.transform(np.vstack(val_Y_raw))

    # 构建窗口 (动态传入W)
    X_tr_win, Y_tr_win = create_windows(train_X_scaled, train_Y_scaled, W)
    X_val_win, Y_val_win = create_windows(val_X_scaled, val_Y_scaled, W)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_win), torch.FloatTensor(Y_tr_win)),
                              batch_size=BATCH_SIZE, shuffle=True)

    # 实例化并训练模型 (动态传入 HIDDEN_DIM 和 LR)
    model = LSTM_Model(input_dim=28, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.HuberLoss()

    model.train()
    for epoch in range(EPOCHS):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    # 推理与评估
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.FloatTensor(X_val_win).to(DEVICE)).cpu().numpy()

    preds_real = scaler_Y.inverse_transform(preds_scaled)
    true_real = scaler_Y.inverse_transform(Y_val_win)

    # 计算提插 R_Z
    if np.std(preds_real[:, 0]) != 0 and np.std(true_real[:, 0]) != 0:
        r_z, _ = pearsonr(preds_real[:, 0], true_real[:, 0])
    else:
        r_z = 0

    # 计算捻转 R_Theta
    if np.std(preds_real[:, 1]) != 0 and np.std(true_real[:, 1]) != 0:
        r_theta, _ = pearsonr(preds_real[:, 1], true_real[:, 1])
    else:
        r_theta = 0

    # 返回综合得分 (可以简单取平均，也可以按需加权)
    return r_z, r_theta


# ================= 3. 运行网格搜索主程序 =================
if __name__ == "__main__":
    print("🚀 开始超参数网格搜索 (Grid Search)...")
    best_score = -1.0
    best_params = {}

    total_combinations = len(SEARCH_SPACE['W']) * len(SEARCH_SPACE['LR']) * \
                         len(SEARCH_SPACE['HIDDEN_DIM']) * len(SEARCH_SPACE['EPOCHS'])
    current = 1

    for w in SEARCH_SPACE['W']:
        for lr in SEARCH_SPACE['LR']:
            for hd in SEARCH_SPACE['HIDDEN_DIM']:
                for e in SEARCH_SPACE['EPOCHS']:
                    print(f"[{current}/{total_combinations}] 测试: W={w}, LR={lr}, Hidden={hd}, Epochs={e} ...", end=" ")

                    r_z, r_theta = train_and_eval(W=w, LR=lr, HIDDEN_DIM=hd, EPOCHS=e)
                    # 计算双轴平均分，作为选拔标准
                    combined_score = (r_z + r_theta) / 2.0
                    print(f"R_Z={r_z:.4f}, R_Theta={r_theta:.4f} => 综合={combined_score:.4f}")
                    # 根据综合得分来更新最佳参数
                    if combined_score > best_score:
                        best_score = combined_score
                        # 记录最佳参数时也加上 EPOCHS
                        best_params = {'W': w, 'LR': lr, 'HIDDEN_DIM': hd, 'EPOCHS': e}

                    current += 1

    print("\n================ 调优完成 ================")
    print(f"🏆 最佳参数组合: {best_params}")
    print(f"🏆 最高得分: {best_score:.4f}")
    print("==========================================")
    print("👉 接下来：请将这些最佳参数填回你主程序的 Config 类中，运行最后的完整留一法验证！")
