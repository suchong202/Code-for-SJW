import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy import signal
import joblib  # 用于保存 .pkl 文件
import warnings

warnings.filterwarnings('ignore')


# ================= 1. 配置最佳参数（请填入你之前网格搜索的最佳值） =================
class Config:
    W = {"Lifting": 15, "Twirling": 15}
    LR = {"Lifting": 0.001, "Twirling": 0.001}
    HIDDEN_DIM = {"Lifting": 128, "Twirling": 128}
    EPOCHS = {"Lifting": 50, "Twirling": 50}
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_PRIOR_FEATURES = True


# --- 把前面的特征预处理函数直接复制过来 ---
def smooth_data(data, span=5):
    return pd.DataFrame(data).ewm(span=span, adjust=False).mean().values


def get_dynamic_shift(v_vision, v_robot, max_shift=10):
    if np.std(v_vision) == 0 or np.std(v_robot) == 0: return 0

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


def load_and_preprocess_trial(hand_path, pos_path, task_type):
    df_hand = pd.read_csv(hand_path)
    hand_raw = smooth_data(df_hand.iloc[:, 1:28].values, span=5)
    N = hand_raw.shape[0]
    nodes = hand_raw.reshape(N, 9, 3)

    wrist_y = -nodes[:, 0, 1:2]
    node_0, node_4, node_8 = nodes[:, 0, :], nodes[:, 4, :], nodes[:, 8, :]
    rel_features = (nodes[:, 1:, :] - nodes[:, 0:1, :]).reshape(N, 24)
    d_pinch = np.linalg.norm(node_8 - node_4, axis=1, keepdims=True)
    v_y = np.zeros((N, 1))
    v_y[1:, 0] = -(nodes[1:, 0, 1] - nodes[:-1, 0, 1])

    rel_thumb = node_4 - node_0
    v_thumb_x = np.zeros((N, 1))
    v_thumb_x[1:, 0] = -(rel_thumb[1:, 0] - rel_thumb[:-1, 0])
    v_thumb_x = smooth_data(v_thumb_x, span=10)

    if Config.USE_PRIOR_FEATURES:
        if task_type == "Lifting":
            X_features = np.concatenate([rel_features, wrist_y, v_y], axis=1)
        elif task_type == "Twirling":
            X_features = np.concatenate([rel_features, v_thumb_x, d_pinch], axis=1)
    else:
        X_features = rel_features

    df_pos = pd.read_csv(pos_path)
    pos_raw = df_pos.iloc[:, 1:3].values
    Y_diff = np.zeros((N, 2))
    Y_diff[1:, :] = pos_raw[1:, :] - pos_raw[:-1, :]

    N_frames = len(X_features)
    if task_type == "Lifting":
        lag_z = get_dynamic_shift(v_y.flatten(), Y_diff[:, 0].flatten(), max_shift=5)
        t_start = max(0, -lag_z)
        length = min(N_frames - t_start, N_frames - (t_start + lag_z))
        X_aligned = X_features[t_start: t_start + length]
        Y_aligned = Y_diff[t_start + lag_z: t_start + length + lag_z, 0:1]
        return X_aligned, Y_aligned
    elif task_type == "Twirling":
        lag_theta = get_dynamic_shift(v_thumb_x.flatten(), Y_diff[:, 1].flatten(), max_shift=10)
        t_start = max(0, -lag_theta)
        length = min(N_frames - t_start, N_frames - (t_start + lag_theta))
        X_aligned = X_features[t_start: t_start + length]
        Y_aligned = Y_diff[t_start + lag_theta: t_start + length + lag_theta, 1:2]
        return X_aligned, Y_aligned


def create_windows(X, Y, W):
    X_win, Y_win = [], []
    for i in range(len(X) - W + 1):
        X_win.append(X[i:i + W, :])
        Y_win.append(Y[i + W - 1, :])
    return np.array(X_win), np.array(Y_win)


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ================= 3. 训练最终模型的函数 =================
def train_final_expert(task_type):
    print(f"\n🚀 开始整合训练【{task_type} 专家】的最终完全体模型...")

    # 提取超参数
    W = Config.W[task_type]
    LR = Config.LR[task_type]
    HIDDEN_DIM = Config.HIDDEN_DIM[task_type]
    EPOCHS = Config.EPOCHS[task_type]

    if task_type == "Lifting":
        styles = [1, 2]
    else:
        styles = [3, 4]
    trials = [1, 2, 3, 4, 5]  # 使用 100% 的数据，不留测试集！

    all_X_raw, all_Y_raw = [], []

    for style in styles:
        for t in trials:
            h_path = f"data/data_raw/data_hand/{style}/{t}.csv"
            p_path = f"data/data_raw/data_pos/{style}/{t}.csv"
            if not os.path.exists(h_path): continue
            X, Y = load_and_preprocess_trial(h_path, p_path, task_type)
            all_X_raw.append(X)
            all_Y_raw.append(Y)

    all_X_raw = np.vstack(all_X_raw)
    all_Y_raw = np.vstack(all_Y_raw)
    input_dim = all_X_raw.shape[1]

    # 💡 核心步骤：拟合并保存 StandardScaler (.pkl 文件)
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    all_X_scaled = scaler_X.fit_transform(all_X_raw)
    all_Y_scaled = scaler_Y.fit_transform(all_Y_raw)

    # 保存 .pkl 文件 (给推理程序用)
    joblib.dump(scaler_X, f"model/scaler_X_{task_type}.pkl")
    joblib.dump(scaler_Y, f"model/scaler_Y_{task_type}.pkl")
    print(f"✅ 已保存特征归一化器: scaler_X_{task_type}.pkl 和 scaler_Y_{task_type}.pkl")

    # 构建窗口
    X_win, Y_win = create_windows(all_X_scaled, all_Y_scaled, W)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_win), torch.FloatTensor(Y_win)),
                              batch_size=Config.BATCH_SIZE, shuffle=True)

    # 实例化并训练
    model = LSTM_Model(input_dim, HIDDEN_DIM).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(Config.DEVICE), batch_y.to(Config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch [{epoch + 1}/{EPOCHS}] Loss: {loss.item():.6f}")

    # 💡 核心步骤：保存 LSTM 模型权重 (.pth 文件)
    torch.save(model.state_dict(), f"model/LSTM_{task_type}_best.pth")
    print(f"✅ 已保存最终模型权重: LSTM_{task_type}_best.pth")


if __name__ == "__main__":
    train_final_expert("Lifting")
    train_final_expert("Twirling")
    print("\n🎉 大功告成！你现在拥有了部署机器人所需的全部文件！")