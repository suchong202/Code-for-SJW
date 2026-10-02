import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy import signal
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ================= 1. 配置区 =================
class Config:
    W = {"Lifting": 15, "Twirling": 15}             # 滑动窗口大小
    LR = {"Lifting": 0.001, "Twirling": 0.001}      # 学习率
    HIDDEN_DIM = {"Lifting": 128, "Twirling": 128}  # 隐藏层维度
    EPOCHS = {"Lifting": 50, "Twirling": 50}        # 训练轮数
    BATCH_SIZE = 32

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 2. 动态特征提取 =================
def load_and_preprocess_trial(hand_path, pos_path, task_type, feature_mode='full'):
    """
    feature_mode 支持三种:
    1. 'skeleton_only': 仅24维相对坐标
    2. 'skeleton_wrist': 24维相对坐标 + 1维手腕绝对坐标 (25维)
    3. 'full': 完整28维 (加入提取的先验特征)
    """
    # 1. 读取视觉数据
    df_hand = pd.read_csv(hand_path)
    hand_raw = df_hand.iloc[:, 1:28].values
    hand_raw = smooth_data(hand_raw, span=5)
    N = hand_raw.shape[0]

    nodes = hand_raw.reshape(N, 9, 3)

    wrist_y = -nodes[:, 0, 1:2]  # 获取手腕Y轴绝对坐标
    # 先验特征
    node_0 = nodes[:, 0, :]
    node_4 = nodes[:, 4, :]
    node_8 = nodes[:, 8, :]

    # 特征组件
    rel_nodes = nodes[:, 1:, :] - nodes[:, 0:1, :]
    rel_features = rel_nodes.reshape(N, 24)

    d_pinch = np.linalg.norm(node_8 - node_4, axis=1, keepdims=True)

    v_y = np.zeros((N, 1))
    v_y[1:, 0] = -(nodes[1:, 0, 1] - nodes[:-1, 0, 1])

    rel_thumb = node_4 - node_0
    v_thumb_x = np.zeros((N, 1))
    v_thumb_x[1:, 0] = -(rel_thumb[1:, 0] - rel_thumb[:-1, 0]) # (加负号看波形情况)
    v_thumb_x = smooth_data(v_thumb_x, span=10)

    # 根据 mode 拼接特征
    if feature_mode == 'skeleton_only':
        X_features = rel_features
    elif feature_mode == 'skeleton_wrist':
        if task_type == "Lifting":
            X_features = np.concatenate([rel_features, wrist_y], axis=1)
        elif task_type == "Twirling":
            X_features = np.concatenate([rel_features, v_thumb_x], axis=1)
    elif feature_mode == 'full':
        if task_type == "Lifting":
            # 提插专家：加入提插专属特征 (剔除捻转特征)
            X_features = np.concatenate([rel_features, wrist_y, v_y], axis=1)
        elif task_type == "Twirling":
            # 捻转专家：加入捻转专属特征 (剔除 wrist_y 和 v_y 干扰)
            X_features = np.concatenate([rel_features, v_thumb_x, d_pinch], axis=1)

    # 2.读取机械臂数据
    df_pos = pd.read_csv(pos_path)
    pos_raw = df_pos.iloc[:, 1:3].values

    Y_diff = np.zeros((N, 2))
    Y_diff[1:, :] = pos_raw[1:, :] - pos_raw[:-1, :]

    # 3.时间对齐
    N_frames = len(X_features)

    if task_type == "Lifting":
        lag_z = get_dynamic_shift(v_y.flatten(), Y_diff[:, 0].flatten(), max_shift=5)
        t_start = max(0, -lag_z)
        length = min(N_frames - t_start, N_frames - (t_start + lag_z))
        X_aligned = X_features[t_start: t_start + length]
        Y_aligned = Y_diff[t_start + lag_z: t_start + length + lag_z, 0:1]  # 只取 Z 轴 (形状 N,1)
        return X_aligned, Y_aligned

    elif task_type == "Twirling":
        lag_theta = get_dynamic_shift(v_thumb_x.flatten(), Y_diff[:, 1].flatten(), max_shift=10)
        t_start = max(0, -lag_theta)
        length = min(N_frames - t_start, N_frames - (t_start + lag_theta))
        X_aligned = X_features[t_start: t_start + length]
        Y_aligned = Y_diff[t_start + lag_theta: t_start + length + lag_theta, 1:2]  # 只取 Theta 轴 (形状 N,1)
        return X_aligned, Y_aligned



def create_windows(X, Y, W):
    X_win, Y_win = [], []
    for i in range(len(X) - W + 1):
        X_win.append(X[i:i + W, :])
        Y_win.append(Y[i + W - 1, :])
    return np.array(X_win), np.array(Y_win)


def smooth_data(data, span=5):
    """EMA 因果滤波"""
    df = pd.DataFrame(data)
    return df.ewm(span=span, adjust=False).mean().values


def get_dynamic_shift(v_vision, v_robot, max_shift=10):
    """带去极值鲁棒处理的动态对齐"""
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



# ================= 3. LSTM 模型 (统一使用LSTM测试特征有效性) =================
class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后时刻的隐藏状态
        return self.fc(out)


def train_dl_model(model, train_loader, lr, epochs, desc="Training"):
    model.to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    # criterion = nn.HuberLoss()

    pbar = tqdm(range(epochs), desc=desc, leave=False, position=1)
    for epoch in pbar:
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(Config.DEVICE), batch_y.to(Config.DEVICE)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{total_loss / len(train_loader):.6f}"})
    return model


def evaluate(pred, true):
    """计算单任务的 RMSE, Pearson, DTW"""
    rmse = np.sqrt(np.mean((pred[:, 0] - true[:, 0]) ** 2))
    r, _ = pearsonr(pred[:, 0], true[:, 0]) if np.std(pred[:, 0]) != 0 and np.std(true[:, 0]) != 0 else (0, 0)
    dtw_val, path = fastdtw(pred[:, 0:1], true[:, 0:1], dist=euclidean)
    return rmse, r, dtw_val / len(path)


# ================= 4. 消融实验循环 =================
def run_ablation(feature_mode, task_type):
    # 根据任务区分手法 Style
    if task_type == "Lifting":
        styles = [1, 2]
    elif task_type == "Twirling":
        styles = [3, 4]

    trials = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # trials = [1, 2, 3, 4, 5]

    # 提取当前任务的最佳超参数
    current_W = Config.W[task_type]
    current_lr = Config.LR[task_type]
    current_hidden = Config.HIDDEN_DIM[task_type]
    current_epochs = Config.EPOCHS[task_type]

    all_results = []
    total_folds = len(styles) * len(trials)
    main_pbar = tqdm(total=total_folds, desc=f"Ablation ({task_type}): {feature_mode}", position=0, leave=True)

    for style in styles:
        for test_trial in trials:
            train_X_raw, train_Y_raw, test_X_raw, test_Y_raw = [], [], [], []
            for t in trials:
                h_path, p_path = f"../data/data_hand/{style}/{t}.csv", f"../data/data_pos/{style}/{t}.csv"
                if not os.path.exists(h_path): continue

                # ==== 核心：根据不同模式提取特征 ====
                X, Y = load_and_preprocess_trial(h_path, p_path, task_type, feature_mode=feature_mode)

                if t == test_trial:
                    test_X_raw.append(X);
                    test_Y_raw.append(Y)
                else:
                    train_X_raw.append(X);
                    train_Y_raw.append(Y)

            if not train_X_raw or not test_X_raw: continue

            train_X_raw, train_Y_raw = np.vstack(train_X_raw), np.vstack(train_Y_raw)
            test_X_raw, test_Y_raw = np.vstack(test_X_raw), np.vstack(test_Y_raw)

            input_dim = train_X_raw.shape[1]

            scaler_X, scaler_Y = StandardScaler(), StandardScaler()
            train_X_scaled = scaler_X.fit_transform(train_X_raw)
            train_Y_scaled = scaler_Y.fit_transform(train_Y_raw)
            test_X_scaled = scaler_X.transform(test_X_raw)
            test_Y_scaled = scaler_Y.transform(test_Y_raw)

            X_tr_win, Y_tr_win = create_windows(train_X_scaled, train_Y_scaled, current_W)
            X_te_win, Y_te_win = create_windows(test_X_scaled, test_Y_scaled, current_W)

            train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_win), torch.FloatTensor(Y_tr_win)),
                                      batch_size=Config.BATCH_SIZE, shuffle=True)

            model = LSTM_Model(input_dim, hidden_dim=current_hidden)

            model = train_dl_model(model, train_loader, lr=current_lr, epochs=current_epochs, desc=f"S{style}_T{test_trial}")

            model.eval()
            with torch.no_grad():
                preds_scaled = model(torch.FloatTensor(X_te_win).to(Config.DEVICE)).cpu().numpy()

            preds_real = scaler_Y.inverse_transform(preds_scaled)
            true_real = scaler_Y.inverse_transform(Y_te_win)

            rmse, r, dtw_val = evaluate(preds_real, true_real)

            if task_type == "Lifting":
                rmse_z, r_z, dtw_z = rmse, r, dtw_val
                rmse_theta, r_theta, dtw_theta = np.nan, np.nan, np.nan
            else:
                rmse_z, r_z, dtw_z = np.nan, np.nan, np.nan
                rmse_theta, r_theta, dtw_theta = rmse, r, dtw_val

            all_results.append({
                "Feature_Mode": feature_mode,
                "Task": task_type,
                "Style": style,
                "Test_Trial": test_trial,
                "RMSE_Z": rmse_z, "RMSE_Theta": rmse_theta,
                "R_Z": r_z, "R_Theta": r_theta,
                "DTW_Z": dtw_z, "DTW_Theta": dtw_theta
            })

            main_pbar.update(1)
            main_pbar.set_postfix({"R": f"{r:.4f}"})

    main_pbar.close()
    return pd.DataFrame(all_results)


if __name__ == "__main__":
    print("--- 准备进行特征消融实验 ---")
    modes = ['skeleton_only', 'skeleton_wrist', 'full']
    tasks = ['Lifting', 'Twirling']

    results = []

    for task in tasks:
        print(f"\n================ 开始 {task} 任务消融实验 ================")
        for m in modes:
            df_res = run_ablation(m, task)
            results.append(df_res)

    final_df = pd.concat(results)
    final_df.to_csv("ablation_results.csv", index=False)
    print("✅ 消融实验全部完成！已保存至 ablation_results.csv")

    # 分别打印直观总结
    for task in tasks:
        print(f"\n[{task} 专家消融总结]")
        task_df = final_df[final_df['Task'] == task]
        for m in modes:
            mode_df = task_df[task_df['Feature_Mode'] == m]
            if task == 'Lifting':
                mean_res = mode_df[["RMSE_Z", "R_Z", "DTW_Z"]].mean()
                print(f"模式: {m:15s} -> RMSE={mean_res['RMSE_Z']:.4f}, R={mean_res['R_Z']:.4f}, DTW={mean_res['DTW_Z']:.4f}")
            else:
                mean_res = mode_df[["RMSE_Theta", "R_Theta", "DTW_Theta"]].mean()
                print(f"模式: {m:15s} -> RMSE={mean_res['RMSE_Theta']:.4f}, R={mean_res['R_Theta']:.4f}, DTW={mean_res['DTW_Theta']:.4f}")

