import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


# ================= 1. 配置与模型定义 =================
class Config:
    # 必须和训练时保持一致
    W = {"Lifting": 15, "Twirling": 15}
    HIDDEN_DIM = {"Lifting": 128, "Twirling": 128}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 机械臂入针的初始姿态 (单位: 米 和 弧度，请根据实际实验修改)
    Z_0 = 0.200
    THETA_0 = 0.0


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def smooth_data(data, span=5):
    """EMA 因果滤波"""
    df = pd.DataFrame(data)
    return df.ewm(span=span, adjust=False).mean().values


# ================= 2. 视觉处理模块 (Video to CSV) =================
def extract_hand_from_video(video_path, output_csv):
    """利用 MediaPipe 按 10Hz 提取视频中的手部关键点"""
    print(f"🎥 正在处理视频 {video_path}...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("无法读取视频帧率，请检查视频文件是否损坏！")

    frame_interval = max(1, int(fps / 10))  # 降采样到 10Hz

    data_list = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_count % frame_interval == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                row = []
                for i in range(9):
                    lm = hand_landmarks.landmark[i]
                    row.extend([lm.x, lm.y, lm.z])
                data_list.append(row)
            else:
                if len(data_list) > 0:
                    data_list.append(data_list[-1])
                else:
                    data_list.append([0.0] * 27)

        frame_count += 1

    cap.release()
    df = pd.DataFrame(data_list)
    df.insert(0, 'Frame', range(1, len(df) + 1))
    df.to_csv(output_csv, index=False)
    print(f"✅ 视频处理完成，提取到 {len(df)} 帧 (10Hz) 骨架数据！")
    return output_csv


# ================= 3. VPM 推理与特征工程模块 =================
def prepare_inference_features(csv_path):
    """推理时的特征工程，与双流架构训练阶段严格一致"""
    df_hand = pd.read_csv(csv_path)
    hand_raw = df_hand.iloc[:, 1:28].values
    hand_raw = smooth_data(hand_raw, span=5)
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

    # 双流特征彻底隔离
    X_lift = np.concatenate([rel_features, wrist_y, v_y], axis=1)  # 提插：27维
    X_twirl = np.concatenate([rel_features, v_thumb_x, d_pinch], axis=1)  # 捻转：26维

    return X_lift, X_twirl


def create_inference_windows(X, W):
    X_win = []
    for i in range(len(X) - W + 1):
        X_win.append(X[i:i + W, :])
    return np.array(X_win)


def run_vpm_inference(video_path, task_type):
    # 1. 提取视觉骨架
    csv_path = extract_hand_from_video(video_path, "temp_inference.csv")
    X_lift_raw, X_twirl_raw = prepare_inference_features(csv_path)

    # 2. 加载训练好的 StandardScaler (注意：这里需要你在训练时用 joblib 保存它们)
    # 例如：joblib.dump(scaler_X_lift, 'scaler_X_lift.pkl')
    print("🧠 正在加载归一化器 (.pkl) 与专家 LSTM 模型 (.pth)...")
    scaler_X = joblib.load(f"model/scaler_X_{task_type}.pkl")
    scaler_Y = joblib.load(f"model/scaler_Y_{task_type}.pkl")

    # 根据 task_type 提取对应的特征
    X_raw = X_lift_raw if task_type == "Lifting" else X_twirl_raw

    # 归一化并创建窗口
    X_scaled = scaler_X.transform(X_raw)
    X_win = create_inference_windows(X_scaled, Config.W[task_type])

    # 3. 实例化并加载对应的专家模型
    model = LSTM_Model(input_dim=X_raw.shape[1], hidden_dim=Config.HIDDEN_DIM[task_type]).to(Config.DEVICE)
    model.load_state_dict(torch.load(f"model/LSTM_{task_type}_best.pth", map_location=Config.DEVICE))
    model.eval()

    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(X_win).to(Config.DEVICE)).cpu().numpy()

    # 4. 反归一化得到真实的物理增量 (ΔZ, Δθ)
    delta_val = scaler_Y.inverse_transform(pred_scaled).flatten()

    return delta_val


# ================= 4. 机械臂绝对轨迹生成与插值模块 =================
def generate_robot_trajectory(delta_val, task_type, input_video_path):
    """ 积分累加 -> 100Hz 高频上采样插值 """
    print("🦾 正在生成 100Hz 机械臂高频控制轨迹({task_type})...")

    # 解析输入路径获取 style 和 trial (兼容 Windows 和 Mac 路径分隔符)
    path_parts = input_video_path.replace('\\', '/').split('/')
    style_dir = path_parts[-2]  # 获取如 "1"
    trial_name = path_parts[-1].split('.')[0]  # 获取如 "1"

    # 自动创建输出文件夹
    out_dir = f"data/2_data_execution_trajectory/{style_dir}"
    os.makedirs(out_dir, exist_ok=True)

    csv_out_path = f"{out_dir}/{trial_name}.csv"
    png_out_path = f"{out_dir}/{trial_name}.png"

    # 1. 积分累加：计算绝对轨迹 (补齐因为时间窗口而丢失的首帧)
    if task_type == "Lifting":
        abs_val = np.cumsum(delta_val)                  # 第一步：先积出漂移的原始轨迹
        abs_val = signal.detrend(abs_val) + Config.Z_0  # 第二步：强行拉平漂移，并平移到初始入针点 Z_0
        col_name = 'Z_Command_m'
        plot_title, ylabel = "Kinematic Tracking Performance in Lifting", "Lifting Depth (mm)"
        plot_val = abs_val * 1000  # 转毫米
    else:
        abs_val = np.cumsum(delta_val)                      # 第一步：积分
        abs_val = signal.detrend(abs_val) + Config.THETA_0  # 第二步：强行拉平漂移，并加回初始角度
        col_name = 'Theta_Command_rad'
        plot_title, ylabel = "Kinematic Tracking Performance in Twirling", "Twirling Angle (°)"
        plot_val = abs_val  # 转角度

    # 2. 三次样条插值：10Hz 提升到 100Hz
    t_10hz = np.arange(len(abs_val)) * 0.1
    t_100hz = np.arange(0, t_10hz[-1], 0.01)
    spline = CubicSpline(t_10hz, abs_val)
    traj_100hz = spline(t_100hz)

    # 3. 保存给机械臂执行的文件
    robot_cmd_df = pd.DataFrame({'Time_s': t_100hz, col_name: traj_100hz})
    # robot_cmd_df.to_csv("2_robot_execution_trajectory.csv", index=False)
    robot_cmd_df.to_csv(csv_out_path, index=False)
    print(f"✅ 连续控制轨迹已保存至: {csv_out_path}")

    # 4. 画图确认
    plt.rcParams['font.family'] = ['Calibri']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 4))
    plt.plot(t_10hz, plot_val, 'o', label='10Hz VPM Prediction Point', alpha=0.5)

    if task_type == "Lifting":
        plt.plot(t_100hz, traj_100hz * 1000, '-', label='100Hz Smooth control trajectory', color='#E64B35')
    else:
        plt.plot(t_100hz, traj_100hz, '-', label='100Hz Smooth control trajectory', color='#3C5488')

    plt.title(plot_title)
    plt.xlabel("时间 (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle=':')

    plt.tight_layout()
    # plt.savefig(f"plot/2_Fig_Robot_Trajectory_{task_type}.png", dpi=300)
    plt.savefig(png_out_path, dpi=300)
    plt.show()


if __name__ == "__main__":

    task_type = 'Lifting'
    # task_type = 'Twirling'

    # 请放一段中医专家操作的视频在此目录下，重命名为 test_hand_video.mp4
    # TEST_VIDEO = "test_hand_video.mp4"
    TEST_VIDEO = "data/1_data_hand/1/1.mp4"

    if os.path.exists(TEST_VIDEO):
        # 将 task_type 传入推理和轨迹生成函数
        delta_val = run_vpm_inference(TEST_VIDEO, task_type)
        generate_robot_trajectory(delta_val, task_type, TEST_VIDEO)
    else:
        print(f"⚠️ 找不到视频文件 {TEST_VIDEO}，请录制一段针灸手势视频放进来！")