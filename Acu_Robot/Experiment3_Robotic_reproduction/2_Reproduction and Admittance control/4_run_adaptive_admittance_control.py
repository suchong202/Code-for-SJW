import sys
import os
import time
import threading
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 确保能找到睿尔曼的 SDK 库
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Robotic_Arm.rm_robot_interface import *


# ================= 1. 核心控制参数配置 =================
class Config:
    ROBOT_IP = "192.168.1.18"
    ROBOT_PORT = 8080
    BASE_POSE = [-0.300, 0.000, 0.300, 3.140, 0.000, 0.000]

    STEP = 2
    LOOP_INTERVAL = 0.02  # 50Hz (20ms)

    # 【导纳控制基础参数 M, B, K - 提插】
    M_Z = 2.0        # 虚拟质量 0.1-3.0     初始：0.5      仿真皮肤：2.0        真实猪肉：2.0
    B_Z_0 = 100.0    # 基础阻尼 20-200      初始：50.0     仿真皮肤：100.0      真实猪肉：100
    K_Z_0 = 1000.0   # 基础刚度 100-1000    初始：200.0    仿真皮肤：1000.0     真实猪肉：1000
    FORCE_DEADZONE = 0.5  # 传感器底噪死区 (N)，小于此力不触发柔顺    初始：200.0    仿真皮肤：0.5     真实猪肉：0.2
    FORCE_SAFE_MAX = 2.0  # 针刺肉类的安全受力上限 (N)，超过则触发模糊退让    初始：200.0    仿真皮肤：2.0     真实猪肉：0.8
    MAX_ADMITTANCE_Z = 0.01  # 最大允许的柔顺退让位移 (1厘米)，防止系统发散失控

    # 【导纳控制基础参数 M, B, K - 捻转】
    M_THETA = 0.2        # 虚拟转动惯量 (kg·m^2) 0.05
    B_THETA_0 = 5.0     # 基础旋转阻尼 (N·m·s/rad) 10.0
    K_THETA_0 = 100.0    # 基础旋转刚度 (N·m/rad) 50.0
    TORQUE_DEADZONE = 0.001  # 扭矩传感器底噪死区 (N·m)
    TORQUE_SAFE_MAX = 0.02  # 安全扭矩上限 (N·m)，遇到肌纤维缠绕的阈值
    MAX_ADMITTANCE_THETA = 0.26  # 最大允许退让角 (约15度)，防止发散

    # 【安全阈值与方向参数】
    # 如果机械臂越扎越深发散，请将 1.0 改为 -1.0
    FORCE_DIR_Z = -1.0
    TORQUE_DIR_Z = -1.0

    MAX_VELOCITY = 0.015  # 最大退让速度限制 (1.5厘米/秒)
    MAX_ACCEL = 1.0  # 最大退让加速度限制


# ================= 2. 模糊推理引擎 (Fuzzy Inference System) =================
class FuzzyAdmittanceController:
    def __init__(self):
        # 1. 定义模糊输入变量
        # E_f: 受力超载误差 (当前力 - 安全力). 负数代表安全，正数代表超载危险
        self.E_f = ctrl.Antecedent(np.arange(-2.0, 2.0, 0.05), 'E_f')
        # EC_f: 力的变化率 (导数). 正数代表力正在急剧变大
        self.EC_f = ctrl.Antecedent(np.arange(-20.0, 20.0, 0.5), 'EC_f')

        # 2. 定义模糊输出变量
        # dK: 刚度调整量. 负数代表变软，正数代表变硬
        self.dK = ctrl.Consequent(np.arange(-500, 100, 5), 'dK')
        # dB: 阻尼调整量. 正数代表增加阻尼(急刹车)
        self.dB = ctrl.Consequent(np.arange(-20, 100, 1), 'dB')

        # 3. 隶属度函数 (Membership Functions) - 简单设为 N(负), Z(零), P(正)
        self.E_f.automf(names=['N', 'Z', 'P'])
        self.EC_f.automf(names=['N', 'Z', 'P'])
        self.dK.automf(names=['N', 'Z', 'P'])
        self.dB.automf(names=['N', 'Z', 'P'])

        # 4. 定义模糊规则库 (核心创新点)
        rules = [
            # 危险情况：受力大(P)且还在增加(P) -> 大幅变软(N)，大幅刹车(P)
            ctrl.Rule(self.E_f['P'] & self.EC_f['P'], (self.dK['N'], self.dB['P'])),
            # 警告情况：受力大(P)但趋于稳定(Z) -> 变软(N)，适度刹车(Z)
            ctrl.Rule(self.E_f['P'] & self.EC_f['Z'], (self.dK['N'], self.dB['Z'])),
            # 安全情况：受力小(N) -> 保持高刚度强力跟随(P)，低阻尼(N)
            ctrl.Rule(self.E_f['N'], (self.dK['P'], self.dB['N'])),
            # 边界情况：平稳(Z) -> 维持现状(Z)
            ctrl.Rule(self.E_f['Z'], (self.dK['Z'], self.dB['Z']))
        ]

        # 5. 构建控制系统
        self.fis = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

    def compute(self, force, last_force, dt, deadzone, safe_max):
        """计算实时调整的 Delta K 和 Delta B"""
        # 计算力学特征
        abs_force = abs(force)
        # 如果力处于死区内，直接返回不调整（保持基础高刚度）
        if abs_force < deadzone:
            return 0.0, 0.0

        error = abs_force - safe_max
        error_change = (abs_force - abs(last_force)) / dt

        # 限制输入范围防越界
        self.fis.input['E_f'] = np.clip(error, -3, 2.9)
        self.fis.input['EC_f'] = np.clip(error_change, -10, 9.9)

        try:
            self.fis.compute()
            return self.fis.output['dK'], self.fis.output['dB']
        except:
            return 0.0, 0.0


# ================= 3. 机械臂与传感器控制器 =================
class RobotArmController:
    def __init__(self, ip, port):
        self.robot = RoboticArm(rm_thread_mode_e(2))
        self.handle = self.robot.rm_create_robot_arm(ip, port, 3)
        if self.handle.id == -1:
            print("❌ 机械臂连接失败！")
            exit(1)

    def disconnect(self):
        self.robot.rm_delete_robot_arm()

    def movej_p(self, pose, v=20, block=1):
        return self.robot.rm_movej_p(pose, v, 0, 0, block)

    def movep_canfd(self, pose, follow=True):
        pose_float = [float(val) for val in pose]
        return self.robot.rm_movep_canfd(pose_float, follow)

    def movej_canfd(self, joint, follow=True):
        joint_float = [float(val) for val in joint]
        return self.robot.rm_movej_canfd(joint_float, follow)

    def get_state_and_force(self):
        """同步读取实际位姿与六维力"""
        z, rz, fz, mz = None, None, 0.0, 0.0
        # 1. 读位姿
        res_p, state = self.robot.rm_get_current_arm_state()
        if res_p == 0 and state is not None:
            try:
                pose = state.get('pose')
                z = pose['position']['z'] if isinstance(pose, dict) else pose[2]
                rz = pose['euler']['rz'] if isinstance(pose, dict) else pose[5]
            except:
                pass
        # 2. 读六维力
        res_f, force_data = self.robot.rm_get_force_data()
        if res_f == 0 and force_data is not None:
            try:
                fz = force_data.get('zero_force_data', [0] * 6)[2]
                mz = force_data.get('zero_force_data', [0] * 6)[5]

                # 如果有时候传感器还没标定清零，可以作为备用读取原始力：
                # fz = force_data.get('force_data', [0]*6)[2]
                # mz = force_data.get('force_data', [0]*6)[5]
            except Exception as e:
                pass
        return z, rz, fz, mz


# ================= 4. 核心力位混合控制循环 =================
def execute_admittance_trajectory(input_csv_path):
    # csv_path = "3_robot_execution_trajectory.csv"
    if not os.path.exists(input_csv_path): return
    df = pd.read_csv(input_csv_path)

    # 解析路径
    path_parts = input_csv_path.replace('\\', '/').split('/')
    style_dir = path_parts[-2]
    trial_name = path_parts[-1].split('.')[0]

    # 1. 自动识别任务类型
    if 'Z_Command_m' in df.columns:
        task_type = "Lifting"
        print("📄 检测到提插轨迹，开启【Z轴 模糊力位混合控制】...")
        origin_val = df['Z_Command_m'].iloc[0]
    elif 'Theta_Command_rad' in df.columns:
        task_type = "Twirling"
        print("📄 检测到捻转轨迹，开启【Rz轴 模糊力矩-位姿混合控制】...")
        origin_val = df['Theta_Command_rad'].iloc[0]
    else:
        return

    robot = RobotArmController(Config.ROBOT_IP, Config.ROBOT_PORT)
    fuzzy_ctrl = FuzzyAdmittanceController()

    print("\n⚠️ 机械臂回安全基准点...")
    robot.movej_p(Config.BASE_POSE, v=20, block=1)
    time.sleep(1)

    # 捻转专属：记录到达初始位置后的基础关节角
    base_joint = None
    if task_type == "Twirling":
        res, state = robot.robot.rm_get_current_arm_state()
        if res == 0: base_joint = list(state['joint'])

    # 导纳状态变量初始化
    x_e = 0.0  # 导纳柔顺位移 (当前)
    v_e = 0.0  # 导纳柔顺速度 (当前)
    last_F = 0.0  # 上一帧受力

    # 数据记录列表 (为了写论文作图！)
    control_logs = []

    # 启动后台传感器读取子线程
    shared_sensor_data = {'z': Config.BASE_POSE[2], 'rz': 0.0, 'fz': 0.0, 'mz': 0.0}
    run_flag = [True]

    def sensor_thread_func():
        while run_flag[0]:
            z, rz, fz, mz = robot.get_state_and_force()
            if z is not None:
                shared_sensor_data.update({'z': z, 'rz': rz, 'fz': fz, 'mz': mz})
            time.sleep(0.005)  # 以极高频率在后台轮询

    sensor_thread = threading.Thread(target=sensor_thread_func)
    sensor_thread.start()

    print(f"\n🚀 开始执行！50Hz 透传控制伴随自适应力反馈...")
    dt = Config.LOOP_INTERVAL
    t_start = time.perf_counter()
    next_control_time = t_start + dt

    for i in range(1, len(df), Config.STEP):
        # 1. 获取 VPM 纯视觉期望轨迹、实时状态与受力
        if task_type == "Lifting":
            vpm_target = Config.BASE_POSE[2] + (df['Z_Command_m'].iloc[i] - origin_val)
            actual_pos = shared_sensor_data['z']
            raw_F = shared_sensor_data['fz'] * Config.FORCE_DIR_Z
            current_F = 0.8 * last_F + 0.2 * raw_F      # 对力信号进行一阶低通滤波 (EMA)，消除毛刺
            M, B0, K0 = Config.M_Z, Config.B_Z_0, Config.K_Z_0
            deadzone, safe_max, max_e = Config.FORCE_DEADZONE, Config.FORCE_SAFE_MAX, Config.MAX_ADMITTANCE_Z
        else:
            vpm_target = Config.BASE_POSE[5] + (df['Theta_Command_rad'].iloc[i] - origin_val)
            actual_pos = shared_sensor_data['rz']
            raw_F = shared_sensor_data['mz'] * Config.FORCE_DIR_Z  # 读 Mz 扭矩
            current_F = 0.8 * last_F + 0.2 * raw_F
            M, B0, K0 = Config.M_THETA, Config.B_THETA_0, Config.K_THETA_0
            deadzone, safe_max, max_e = Config.TORQUE_DEADZONE, Config.TORQUE_SAFE_MAX, Config.MAX_ADMITTANCE_THETA

        # 2. 模糊推理在线调整 K 和 B (外环)
        # 严格限制模糊控制的激活条件: 只有受力逼近安全上限(比如达到安全上限的 60% 时)，才允许模糊引擎变软！
        # if abs(current_F) < (safe_max * 0.5):
            # dK, dB = 0.0, 0.0  # 绝对安全区：保持原始硬度，禁止瞎跳！
        # else:
        dK, dB = fuzzy_ctrl.compute(current_F, last_F, dt, deadzone, safe_max)

        K_current = max(K0 * 0.2, K0 + dK)  # 允许刚度降到20%
        B_current = max(B0 * 0.5, B0 + dB)
        # 临界阻尼保护
        # min_safe_B = 2.0 * np.sqrt(M * K_current)
        # B_current = max(B0 + dB, min_safe_B)

        # 3. 受力死区与虚拟动能耗散
        if abs(current_F) < deadzone:
            F_admittance_input = 0.0
            # 当离开软组织(不受力)时，系统应安静地被弹簧 K 拉回原位。
            # 为了抹除之前累积的震荡动能，强行让虚拟速度每帧衰减 80% (如同进入了高粘滞液体)
            v_e = v_e * 0.2
        else:
            # 扣除死区后的“净超载力”，保证力的连续性，防阶跃冲击
            F_admittance_input = (abs(current_F) - deadzone) * np.sign(current_F)

        # 4. 计算离散导纳模型方程: M*a + B*v + K*x = F_ext (内环)
        # a = (F_ext - B*v - K*x) / M
        acc_e = (F_admittance_input - B_current * v_e - K_current * x_e) / M
        # 【新增：绝对安全防抖限幅 1】限制极限加速度，防止冲击
        acc_e = np.clip(acc_e, -Config.MAX_ACCEL, Config.MAX_ACCEL)

        v_e += acc_e * dt
        # 【新增：绝对安全防抖限幅 2】限制最大退让速度，防止疯狂抽搐
        v_e = np.clip(v_e, -Config.MAX_VELOCITY, Config.MAX_VELOCITY)

        x_e += v_e * dt
        # 【新增：绝对安全防抖限幅 3】位置退让限幅 (原本已有，保持即可)
        x_e = np.clip(x_e, -max_e, max_e)

        # 5. 生成最终力位混合控制指令
        # 最终位置 = VPM期望位置 + 导纳柔顺偏移量
        final_cmd = vpm_target + x_e

        # 下发指令 (解耦：提插走位姿，捻转走关节)
        if task_type == "Lifting":
            target_pose = Config.BASE_POSE.copy()
            target_pose[2] = final_cmd
            robot.movep_canfd(target_pose, follow=True)
        else:
            target_joint = base_joint.copy()
            # 关节增量 = 最终绝对角度 - 基准角度
            target_joint[5] = base_joint[5] + (final_cmd - Config.BASE_POSE[5])
            robot.movej_canfd(target_joint, follow=True)

        # 6. 记录数据用于论文绘图
        t_now = time.perf_counter() - t_start
        # 统一使用泛化列名保存，方便画图
        control_logs.append([t_now, vpm_target, final_cmd, actual_pos, current_F, K_current, B_current])
        last_F = current_F

        # 7. 控制台实时监控打印
        if i % 5 == 0:
            print(
                f"\r[监控] 受力:{current_F:5.2f} | 刚度K:{K_current:5.1f} | 目标:{vpm_target:.4f} | 修正:{final_cmd:.4f}  ", end="", flush=True)

        # 8. 严格时钟休眠
        now = time.perf_counter()
        if now < next_control_time:
            time.sleep(next_control_time - now)
            next_control_time += dt
        else:
            next_control_time = now + dt

    print("\n\n🛑 轨迹执行结束，关闭透传...")
    if task_type == "Lifting":
        robot.movep_canfd(target_pose, follow=False)
    else:
        robot.movej_canfd(target_joint, follow=False)
    time.sleep(0.5)

    # 关闭子线程
    run_flag[0] = False
    sensor_thread.join()
    robot.disconnect()

    # ================= 5. 保存极具价值的控制数据 =================
    df_logs = pd.DataFrame(control_logs, columns=[
        'Time_s', 'VPM_Target', 'Admittance_Target', 'Actual_Pos', 'Force_Torque', 'Adaptive_K', 'Adaptive_B'
    ])

    # 自动创建 4_data_admittance_control_result 输出文件夹
    out_dir = f"data/4_data_admittance_control_result/{style_dir}"
    # out_dir = f"4_admittance_control_results_{task_type}.csv"
    os.makedirs(out_dir, exist_ok=True)
    save_filename = f"{out_dir}/{trial_name}.csv"

    df_logs.to_csv(save_filename, index=False)
    print(f"✅ 力位混合控制数据已保存至 {save_filename}")


if __name__ == "__main__":
    # 在这里直接指定你要读取的文件
    INPUT_CSV = "data/2_data_execution_trajectory/1/10.csv"
    execute_admittance_trajectory(INPUT_CSV)



"""
参数怎么调？（黄金调参指南）
这几个参数直接决定了你的实验曲线漂不漂亮。调参原则是：先定方向，再调 MBK，最后调模糊阈值。

1. 基础物理参数（决定力控的手感）
M_Z (虚拟质量)：系统的“迟钝度/惯性”
意义：M 越大，机械臂起步越慢，但对高频噪声（毛刺力）的过滤能力极强。
怎么调： 不能太小，建议保持 0.1 ~ 3.0 之间。越小机械臂对力的反应越灵敏，太大容易导致反应迟钝。
K_Z_0 (基础刚度)：系统的“倔强程度”
意义：K 越大，机械臂越硬，轨迹跟踪越准；K 越小，机械臂越软，越容易被力推开。
关系：稳态误差 x=F/K。如果希望受 2N 力时退让 5mm，则 K=2/0.005=400。
怎么调：范围 100 ~ 1000。数值越大，代表弹簧越硬。在没有受到危险力时，机械臂需要较高的 K 才能紧紧跟住 VPM 的轨迹。建议设为 200~500。
B_Z_0 (基础阻尼)：系统的“刹车片/粘滞度”
意义：专门用来吸收动能、消除抖动。
关系：必须满足 B ≥ 2√M⋅K！例如 M=2,K=500，则 B 必须大于 2√1000 ≈ 63。如果设为 100，系统将极度稳定丝滑。
怎么调：这是消除抖动最重要的参数！ 范围 20 ~ 200。如果你用手指轻轻推机械臂，它退让后还在原地“荡秋千（来回晃）”，说明阻尼太小了，赶紧把 B 加大！如果它动得像陷入了泥潭，把 B 减小。

2. 安全阈值参数（决定什么时候触发柔顺）
FORCE_DEADZONE (死区)：当前 0.5。
怎么调：看一下传感器在空气里的读数跳动范围。如果空气里总是乱跳 0.3N，就把死区设为 0.5N。屏蔽空气噪声。
FORCE_SAFE_MAX (安全上限)：当前 3.0。
怎么调：这是你论文里最重要的阈值！ 比如猪肉实验中，你希望穿刺力绝不能超过 4 牛顿。就把这个设为 3.5。当力达到 3.5N 时，模糊控制器就会疯狂降低刚度 K，强制机械臂退让。

3. 限幅参数（保命参数）
MAX_ADMITTANCE_Z (最大退让距离)：当前 0.01 (1厘米)。
怎么调：限制机械臂最多只能偏离 VPM 轨迹 1 厘米。如果设太大，机械臂可能会一直往后退直到拔出肉块。
MAX_VELOCITY / MAX_ACCEL：
就是刚才救火加的代码，限制退让速度，防抖。

完整的调参流程 (Workflow)
(1) 设死区（Deadzone）：观察图 1，肉在正常摩擦时力是 1N 左右。把死区设为 1.2 N。低于 1.2N 的力，强制归零，保证轨迹 100% 跟随！
(2) 定阈值（Safe Max）：确定多大的力算危险？假设 2.5 N。
(3) 算 M, B, K：先设 M=2.0，K0=800（保证平时够硬）。计算出临界阻尼 2√1600=80。把 B0 设为 120（过阻尼，防抖）。
"""
