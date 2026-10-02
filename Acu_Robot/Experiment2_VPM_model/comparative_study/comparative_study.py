import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
from scipy import signal
from tqdm import tqdm  # 导入进度条库
import time
import warnings

warnings.filterwarnings('ignore')


# ================= 1. 超参数与配置 =================
class Config:
    W = {"Lifting": 15, "Twirling": 15}              # 滑动窗口大小
    LR = {"Lifting": 0.001, "Twirling": 0.001}       # 学习率
    HIDDEN_DIM = {"Lifting": 128, "Twirling": 128}   # 隐藏层维度
    EPOCHS = {"Lifting": 50, "Twirling": 50}         # 训练轮数
    BATCH_SIZE = 32

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实验标志
    USE_PRIOR_FEATURES = True  # 设置为False即可进行消融实验（仅用骨架特征）


# ================= 2. 数据加载与特征工程 =================
def load_and_preprocess_trial(hand_path, pos_path, task_type):
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

    # 根据 mode 拼接特征
    if Config.USE_PRIOR_FEATURES:
        if task_type == "Lifting":
            X_features = np.concatenate([rel_features, wrist_y, v_y], axis=1)
        elif task_type == "Twirling":
            X_features = np.concatenate([rel_features, v_thumb_x, d_pinch], axis=1)
    else:
        X_features = rel_features  # 24 维 (消融实验：仅骨架)

    # 2. 读取位姿数据
    df_pos = pd.read_csv(pos_path)
    pos_raw = df_pos.iloc[:, 1:3].values # [Z, theta]

    # 转换为增量 (差分)
    Y_diff = np.zeros((N, 2))
    Y_diff[1:, :] = pos_raw[1:, :] - pos_raw[:-1, :]

    # 3.时间对齐
    N_frames = len(X_features)

    if task_type == "Lifting":
        # 1.动态获取提插滞后
        lag_z = get_dynamic_shift(v_y.flatten(), Y_diff[:, 0].flatten(), max_shift=5)

        # 2.计算安全的起始点和长度防越界
        t_start = max(0, -lag_z)
        # 确保 X 和 Y 的切片都不会超过 N_frames
        length = min(N_frames - t_start, N_frames - (t_start + lag_z))

        # 3.切片
        X_aligned = X_features[t_start: t_start + length]
        Y_aligned = Y_diff[t_start + lag_z: t_start + length + lag_z, 0:1]  # 只取 Z 轴 (形状 N,1)

        return X_aligned, Y_aligned

    elif task_type == "Twirling":
        # 1.动态获取捻转滞后
        lag_theta = get_dynamic_shift(v_thumb_x.flatten(), Y_diff[:, 1].flatten(), max_shift=10)

        # 2.计算安全的起始点和长度防越界
        t_start = max(0, -lag_theta)
        # 确保 X 和 Y 的切片都不会超过 N_frames
        length = min(N_frames - t_start, N_frames - (t_start + lag_theta))

        # 3.切片
        X_aligned = X_features[t_start: t_start + length]
        Y_aligned = Y_diff[t_start + lag_theta: t_start + length + lag_theta, 1:2]  # 只取 Theta 轴 (形状 N,1)

        return X_aligned, Y_aligned


def create_windows(X, Y, W):
    """构建时间滑动窗口"""
    X_win, Y_win = [], []
    for i in range(len(X) - W + 1):
        X_win.append(X[i:i + W, :])
        Y_win.append(Y[i + W - 1, :])  # Y取窗口最后一帧的对应标签
    return np.array(X_win), np.array(Y_win)

"""
def smooth_data(data, window_size):
    # 简单的滑动平均滤波
    # return pd.DataFrame(data).rolling(window=window_size, min_periods=1, center=True).mean().values
    # Savitzky-Golay 滤波器
    # return savgol_filter(data, window_length=7, polyorder=2, axis=0)
"""
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


# ================= 3. 深度学习模型定义 =================
class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)  # Linear 输出，无激活函数

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后时刻的隐藏状态
        return self.fc(out)


class RNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNN_Model, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class MLP_Model(nn.Module):
    def __init__(self, input_dim, W, hidden_dim):
        super(MLP_Model, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim * W, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten (Batch, W * Features)
        return self.fc_layers(x)


# ================= 4. 模型训练与评估流程 =================
def train_dl_model(model, train_loader, lr, epochs, desc="Training"):
    model.to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # 增加权重衰减(L2正则化)
    criterion = nn.MSELoss()
    # criterion = nn.HuberLoss() # 使用 HuberLoss 提高对噪声的鲁棒性

    # 使用 tqdm 显示 Epoch 进度
    pbar = tqdm(range(epochs), desc=desc, position=1, leave=False)
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


# ================= 5. 核心循环：留一交叉验证 =================
def run_experiment(model_type="LSTM", task_type="Twirling"):
    # 如果是提插专家，只取 1,2；如果是捻转专家，只取 3,4
    if task_type == "Lifting":
        styles = [1, 2]
        target_idx = 0  # 提插对应 Y_diff 的第 0 列
    elif task_type == "Twirling":
        styles = [3, 4]
        target_idx = 1  # 捻转对应 Y_diff 的第 1 列

    trials = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # trials = [1, 2, 3, 4, 5]

    # 提取当前任务的最佳超参数
    current_W = Config.W[task_type]
    current_lr = Config.LR[task_type]
    current_hidden = Config.HIDDEN_DIM[task_type]
    current_epochs = Config.EPOCHS[task_type]

    # input_dim = 28 if Config.USE_PRIOR_FEATURES else 25 # 24(rel) + 1(wrist_z) + 3(prior)

    all_results = []

    print(f"\n🚀 开始 {model_type} 模型的留一交叉验证评估...")

    # 计算总折数 (20折)
    total_folds = len(styles) * len(trials)
    main_pbar = tqdm(total=total_folds, desc=f"Evaluating {model_type}", position=0, leave=True)

    for style in styles:
        for test_trial in trials:
            train_X_raw, train_Y_raw = [], []
            test_X_raw, test_Y_raw = [], []

            # 读取数据划分训练测试集
            for t in trials:
                # 注意：请根据你实际的相对路径修改以下字符串格式
                h_path = f"../data/data_hand/{style}/{t}.csv"
                p_path = f"../data/data_pos/{style}/{t}.csv"

                # 若文件不存在跳过 (方便你用假数据测试时防止报错)
                if not os.path.exists(h_path): continue

                X, Y = load_and_preprocess_trial(h_path, p_path, task_type)

                if t == test_trial:
                    test_X_raw.append(X)
                    test_Y_raw.append(Y)
                else:
                    train_X_raw.append(X)
                    train_Y_raw.append(Y)

            if len(train_X_raw) == 0 or len(test_X_raw) == 0: continue

            # 合并当前折的训练集
            train_X_raw = np.vstack(train_X_raw)
            train_Y_raw = np.vstack(train_Y_raw)
            test_X_raw = np.vstack(test_X_raw)
            test_Y_raw = np.vstack(test_Y_raw)

            input_dim = train_X_raw.shape[1]

            # 💡 必须在训练集上Fit Scaler，防止数据穿越
            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()

            train_X_scaled = scaler_X.fit_transform(train_X_raw)
            train_Y_scaled = scaler_Y.fit_transform(train_Y_raw)
            test_X_scaled = scaler_X.transform(test_X_raw)
            test_Y_scaled = scaler_Y.transform(test_Y_raw)

            # 构建滑动窗口
            X_tr_win, Y_tr_win = create_windows(train_X_scaled, train_Y_scaled, current_W)
            X_te_win, Y_te_win = create_windows(test_X_scaled, test_Y_scaled, current_W)

            # ----- 训练与推理 -----
            if model_type in ["LSTM", "RNN", "MLP"]:
                train_data = TensorDataset(torch.FloatTensor(X_tr_win), torch.FloatTensor(Y_tr_win))
                val_data = TensorDataset(torch.FloatTensor(X_te_win), torch.FloatTensor(Y_te_win))
                train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, shuffle=False)

                if model_type == "LSTM":
                    model = LSTM_Model(input_dim, hidden_dim=current_hidden)
                elif model_type == "RNN":
                    model = RNN_Model(input_dim, hidden_dim=current_hidden)
                elif model_type == "MLP":
                    model = MLP_Model(input_dim, W=current_W, hidden_dim=current_hidden)

                model = train_dl_model(model, train_loader, lr=current_lr, epochs=current_epochs, desc=f"S{style}_T{test_trial}")

                # 推理
                model.eval()
                with torch.no_grad():
                    preds_scaled = model(torch.FloatTensor(X_te_win).to(Config.DEVICE)).cpu().numpy()


            elif model_type == "SVR":
                X_tr_flat = X_tr_win.reshape(X_tr_win.shape[0], -1)
                X_te_flat = X_te_win.reshape(X_te_win.shape[0], -1)
                svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                svr.fit(X_tr_flat, Y_tr_win.ravel())  # .ravel() 转为一维
                preds_scaled = svr.predict(X_te_flat).reshape(-1, 1)  # reshape 保证反归一化维度正确

            # 反归一化还原为实际物理量级
            preds_real = scaler_Y.inverse_transform(preds_scaled)
            true_real = scaler_Y.inverse_transform(Y_te_win)

            # 对预测的增量曲线进行一次轻量级平滑，能显著降低 DTW
            # preds_real[:, 0] = smooth_data(preds_real[:, 0], span=3).flatten()  # 平滑提插
            # preds_real[:, 1] = smooth_data(preds_real[:, 1], span=3).flatten()  # 平滑捻转

            # 保存画图数据 (加上 task_type 区分文件名)
            if style in [1, 3] and test_trial == 1:
                if model_type == "LSTM":
                    np.save(f"LSTM_{task_type}_preds.npy", preds_real)
                    np.save(f"gt_{task_type}.npy", true_real)
                elif model_type == "MLP":
                    np.save(f"MLP_{task_type}_preds.npy", preds_real)

            # 评估 (根据任务分配结果，没测的填 NaN)
            rmse, r, dtw_val = evaluate(preds_real, true_real)

            if task_type == "Lifting":
                rmse_z, r_z, dtw_z = rmse, r, dtw_val
                rmse_theta, r_theta, dtw_theta = np.nan, np.nan, np.nan
            else:
                rmse_z, r_z, dtw_z = np.nan, np.nan, np.nan
                rmse_theta, r_theta, dtw_theta = rmse, r, dtw_val

            all_results.append({
                "Model": model_type, "Task": task_type, "Style": style, "Test_Trial": test_trial,
                "RMSE_Z": rmse_z, "RMSE_Theta": rmse_theta,
                "R_Z": r_z, "R_Theta": r_theta,
                "DTW_Z": dtw_z, "DTW_Theta": dtw_theta
            })

            # 更新主进度条
            main_pbar.update(1)
            # 在进度条右侧实时显示最近的 R 值
            main_pbar.set_postfix({"Last_R": f"{r:.4f}"})

    main_pbar.close()

    # 分别打印提插或捻转结果
    df_results = pd.DataFrame(all_results)
    if not df_results.empty:
        if task_type == "Lifting":
            mean_res = df_results[["RMSE_Z", "R_Z", "DTW_Z"]].mean()
            print(
                f"[{model_type}-提插专家] RMSE={mean_res['RMSE_Z']:.4f}, R={mean_res['R_Z']:.4f}, DTW={mean_res['DTW_Z']:.4f}")
        else:
            mean_res = df_results[["RMSE_Theta", "R_Theta", "DTW_Theta"]].mean()
            print(
                f"[{model_type}-捻转专家] RMSE={mean_res['RMSE_Theta']:.4f}, R={mean_res['R_Theta']:.4f}, DTW={mean_res['DTW_Theta']:.4f}")
    return df_results


# ================= 主函数 =================
if __name__ == "__main__":
    print("--- 针灸多模态视觉-位姿映射验证实验 ---")

    # 你可以依次运行四种模型来填补论文中的表格
    models_to_test = ["LSTM", "RNN", "MLP", "SVR"]
    tasks_to_test = ["Lifting", "Twirling"]  # 增加任务列表

    all_experiments = []

    for task in tasks_to_test:
        print(f"\n================ 开始训练 {task} 专家模型 ================")
        for m in models_to_test:
            res = run_experiment(model_type=m, task_type=task)
            all_experiments.append(res)

    # 保存结果到CSV用于画箱线图等
    if all_experiments[0] is not None and not all_experiments[0].empty:
        final_df = pd.concat(all_experiments)
        final_df.to_csv("experiment_results.csv", index=False)
        print("\n✅ 所有结果已保存至 experiment_results.csv，可用于在论文中绘制箱线图与填充表格。")

