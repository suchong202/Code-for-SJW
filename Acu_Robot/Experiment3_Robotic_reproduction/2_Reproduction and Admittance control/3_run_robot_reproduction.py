import sys
import os
import time
import pandas as pd
import numpy as np
import threading

# 睿尔曼的 SDK 库
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Robotic_Arm.rm_robot_interface import *

# ================= 1. 机械臂控制参数配置 =================
class Config:
    ROBOT_IP = "192.168.1.18"
    ROBOT_PORT = 8080

    # 严格锁定的安全初始基准位姿 [x, y, z, rx, ry, rz]
    # x=-0.3m, y=0, z=0.3m, rx=3.14(法兰向下)
    BASE_POSE = [-0.300, 0.000, 0.300, 3.140, 0.000, 0.000]

    # 【透传核心参数】
    # 你的 CSV 是 100Hz(0.01s)，我们降频到 50Hz 发送 (与 C++ 保持绝对一致)
    STEP = 2
    LOOP_INTERVAL = 0.02   # 50Hz = 20ms


# ================= 2. 机械臂控制器封装 =================
class RobotArmController:
    def __init__(self, ip, port, level=3, mode=2):
        self.thread_mode = rm_thread_mode_e(mode)
        self.robot = RoboticArm(self.thread_mode)
        self.handle = self.robot.rm_create_robot_arm(ip, port, level)

        if self.handle.id == -1:
            print("❌ 机械臂连接失败，请检查 IP 和网络！")
            exit(1)
        else:
            print(f"✅ 成功连接到机械臂，句柄 ID: {self.handle.id}")

    def disconnect(self):
        handle = self.robot.rm_delete_robot_arm()
        if handle == 0:
            print("✅ 成功断开机械臂连接。")

    def movej_p(self, pose, v=20, r=0, connect=0, block=1):
        """关节空间规划的位姿移动 (用于初始安全回零)"""
        return self.robot.rm_movej_p(pose, v, r, connect, block)

    def movep_canfd(self, pose, follow=True):
        """
        高频透传位姿控制
        :param pose: 目标位姿 [x, y, z, rx, ry, rz]
        :param follow: True 表示持续跟随，False 表示退出透传模式
        """
        # 注意：此处将 Python 的 list 转换为 C 能识别的 float 列表，确保万无一失
        pose_float = [float(val) for val in pose]
        return self.robot.rm_movep_canfd(pose_float, follow)

    def movej_canfd(self, joint, follow=True):
        """关节空间的高频透传控制 (专门用于防死锁捻转)"""
        joint_float = [float(val) for val in joint]
        return self.robot.rm_movej_canfd(joint_float, follow)

    def get_current_state(self):
        """获取并解析机械臂当前的真实位姿"""
        res, state = self.robot.rm_get_current_arm_state()
        if res == 0 and state is not None:
            try:
                pose = state.get('pose')
                if isinstance(pose, dict):
                    z = pose['position']['z']
                    roll = pose['euler']['rz']
                else:
                    z = pose[2]
                    roll = pose[5]
                return z, roll
            except Exception:
                pass
        return None, None


# ================= 3. 主控制流程 =================
def execute_trajectory(input_csv_path):
    # 1. 读取并识别轨迹文件
    # csv_path = "2_robot_execution_trajectory.csv"
    if not os.path.exists(input_csv_path):
        print(f"❌ 找不到轨迹文件: {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path)

    # 解析路径
    path_parts = input_csv_path.replace('\\', '/').split('/')
    style_dir = path_parts[-2]
    trial_name = path_parts[-1].split('.')[0]

    if 'Z_Command_m' in df.columns:
        task_type = "Lifting"
        print(f"📄 检测到任务：【提插手法】，将严格限制在 Z 轴运动。")
        origin_val = df['Z_Command_m'].iloc[0]
    elif 'Theta_Command_rad' in df.columns:
        task_type = "Twirling"
        print(f"📄 检测到任务：【捻转手法】，将严格限制在 Rz 轴旋转。")
        origin_val = df['Theta_Command_rad'].iloc[0]
    else:
        print("❌ CSV 中缺少目标列名！")
        return

    # 2. 连接机械臂并回到安全基准点
    robot = RobotArmController(Config.ROBOT_IP, Config.ROBOT_PORT)

    print("\n⚠️ 警告：机械臂即将移动到初始安全基准点...")
    time.sleep(2)

    # 初始回零速度设置得慢一点 (20%)，保证安全
    res = robot.movej_p(Config.BASE_POSE, v=20, block=1)
    if res != 0:
        print(f"❌ 移动到基准位置失败，错误码：{res}")
        robot.disconnect()
        return

    print("✅ 已到达初始位置，开启 CANFD 透传通道！")
    time.sleep(1)

    # 到达基准位置后，读取基础关节角
    res, state = robot.robot.rm_get_current_arm_state()
    if res == 0:
        base_joint = list(state['joint'])
    else:
        print("❌ 获取基准关节角失败！")
        robot.disconnect()
        return

    # 3. 预处理阶段，离线生成所有点位，降低循环内的算力损耗
    cmds_to_send = []
    print("⏳ 正在预编译 50Hz 控制指令缓存队列...")

    for i in range(0, len(df), Config.STEP):
        if task_type == "Lifting":
            # 提插：使用笛卡尔位姿 [x,y,z,rx,ry,rz]
            target = Config.BASE_POSE.copy()
            delta = df['Z_Command_m'].iloc[i] - origin_val
            target[2] = max(0.100, min(Config.BASE_POSE[2] + delta, 0.600))  # 限幅
            cmds_to_send.append(target)
        else:
            # 捻转：使用关节角度 [j1, j2, j3, j4, j5, j6]
            target = base_joint.copy()
            delta = df['Theta_Command_rad'].iloc[i] - origin_val
            target[5] = base_joint[5] + delta  # 仅改变第6个关节
            cmds_to_send.append(target)

    # 新增：定义后台监控子线程
    run_flag = [True]  # 控制子线程启停的开关
    actual_records = []  # 保存实际轨迹
    target_records = []  # 保存目标轨迹

    # 设定全局绝对起点时间
    t_start_main = time.perf_counter()

    def monitor_thread_func():
        """后台高频读取机械臂状态"""
        while run_flag[0]:
            z, rz = robot.get_current_state()
            if z is not None:
                t_now = time.perf_counter() - t_start_main
                actual_records.append([t_now, z, rz])
            time.sleep(0.015)  # 稍微快于50Hz去轮询，确保数据密度充足，不占用主线程资源

    # 启动监控子线程
    monitor_thread = threading.Thread(target=monitor_thread_func)
    monitor_thread.start()

    # 4. 连续轨迹执行与实时监控循环
    print(f"\n🚀 开始执行！严格锁定 50Hz (20ms) 频率发送...")

    # 设置绝对时钟起点
    next_control_time = time.perf_counter() + Config.LOOP_INTERVAL

    for idx, cmd in enumerate(cmds_to_send):
        # 1. 记录当前要发送的目标指令及时间
        t_target = time.perf_counter() - t_start_main
        target_records.append([t_target, cmd[2], cmd[5]])  # [Time, Target_Z, Target_Rz]

        # 2. 主线程：发送透传指令 (非阻塞极速模式)
        if task_type == "Lifting":
            robot.movep_canfd(cmd, follow=True)  # 提插用位姿透传
        else:
            robot.movej_canfd(cmd, follow=True)  # 捻转用关节透传

        # 3. 打印简易进度条 (使用 \r 同行刷新，不读真实状态防卡顿)
        total_points = len(cmds_to_send)
        if idx % 5 == 0 or idx == total_points - 1:
            # 使用 total_points 进行数学运算
            progress = (idx / total_points) * 100
            print(f"\r正在执行... 进度: [{idx:04d}/{total_points:04d}] {progress:5.1f}%", end="", flush=True)

        # 4. 严格的高精度休眠 (相当于 C++ 的 sleep_until)
        now = time.perf_counter()
        if now < next_control_time:
            time.sleep(next_control_time - now)
            next_control_time += Config.LOOP_INTERVAL
        else:
            # 如果系统卡顿超时，放弃补偿，重新校准时间基准
            next_control_time = now + Config.LOOP_INTERVAL

    print("\n\n🛑 执行结束，正在退出透传模式...")
    # 发送最后一条指令，关闭透传跟随
    if task_type == "Lifting":
        robot.movep_canfd(cmds_to_send[-1], follow=False)
    else:
        robot.movej_canfd(cmds_to_send[-1], follow=False)
    time.sleep(0.5)

    # 合并数据并导出同一张 Excel/CSV 表
    run_flag[0] = False
    monitor_thread.join()  # 等待子线程安全结束

    # 转换为 DataFrame
    df_target = pd.DataFrame(target_records, columns=['Time_s', 'Target_Z_m', 'Target_Theta_rad'])
    df_actual = pd.DataFrame(actual_records, columns=['Time_s', 'Actual_Z_m', 'Actual_Theta_rad'])

    # 利用 merge_asof 按时间就近对齐 (以 Target 的时间戳为准)
    df_merged = pd.merge_asof(df_target, df_actual, on='Time_s', direction='nearest')

    # 导出文件 (CSV 格式，Excel可直接完美打开)
    # out_dir = f"3_robot_tracking_result_{task_type}.csv"
    out_dir = f"data/3_data_tracking_result/{style_dir}"
    os.makedirs(out_dir, exist_ok=True)
    save_filename = f"{out_dir}/{trial_name}.csv"

    df_merged.to_csv(save_filename, index=False)
    print(f"✅ 目标轨迹与真实轨迹已对齐并合并，保存至: {save_filename}")
    # ==============================================================


    # 5. 断开连接
    robot.disconnect()


if __name__ == "__main__":
    # 在这里直接指定你要读取的文件
    # INPUT_CSV = "2_robot_execution_trajectory.csv"
    INPUT_CSV = "data/2_data_execution_trajectory/1/10.csv"
    execute_trajectory(INPUT_CSV)
