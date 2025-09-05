import numpy as np
from controller import Robot, Supervisor,Connector
from ikpy.chain import Chain
import pandas as pd
import numpy as np
from gymnasium import Env
from gymnasium import spaces
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.transform import Rotation as R #如果需要將旋轉矩陣轉換為 四元數 (quaternion) 或 歐拉角 (Euler angles)，可以使用
from collections import defaultdict
from collections import deque
import pandas as pd
import random
import psutil
import os

#-----------------------------------------------------------------------------
#成功率追蹤器
class SuccessTracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.successes = deque(maxlen=window_size)

    def record(self, is_success: bool):
        """記錄每個 episode 的成功狀態"""
        self.successes.append(is_success)

    def get_success_rate(self):
        """計算最近 N 回合的成功比例"""
        if not self.successes:
            return 0.0
        return sum(self.successes) / len(self.successes)

tracker = SuccessTracker(window_size=500)
#----------------------------------------------------------------------------
robot_chain = Chain.from_urdf_file("ur5e.urdf")

#利用順項運動學計算出末端軸位置
def get_endpoint_position(angles):#input為機器手臂各軸馬達的位置(角度，以弧度表示)
    endpoint_position=robot_chain.forward_kinematics(angles)
    return endpoint_position #output為手臂末端軸位置(以轉至矩陣表示)

def rotation_matrix_to_euler_angles(R):#將旋轉矩陣轉換為歐拉角
    """
    輸入為3x3的旋轉矩陣,輸出歐拉角表示法的rx,ry,rz
    Convert a rotation matrix to Euler angles (in degrees) with ZYX order.
    
    :param R: np.ndarray, a 3x3 rotation matrix.
    :return: tuple of Euler angles (rx, ry, rz) in degrees.
    """
    # Check for valid rotation matrix
    if not np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("Invalid rotation matrix")
    
    # Extract angles
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0
    
    # Convert radians to degrees
    rx = math.degrees(rx)
    ry = math.degrees(ry)
    rz = math.degrees(rz)
    
    return rx, ry, rz

def calculate_A_prime(R_BA, t_BA, R_B_prime, t_B_prime):#輸入工件路徑採樣點研磨時的座標平移與歐拉角，輸出工件座標研磨時的平移與歐拉角
    """
    根据 B' 座标系在世界座标系下的表示，计算 A' 在世界座标系下的表示。

    :param R_BA: np.ndarray, B 座标系在 A 座标系下的旋转矩阵 (3x3).
    :param t_BA: np.ndarray, B 座标系在 A 座标系下的平移向量 (3x1).
    :param R_B_prime: np.ndarray, B' 座标系在世界座标系下的旋转矩阵 (3x3).
    :param t_B_prime: np.ndarray, B' 座标系在世界座标系下的平移向量 (3x1).
    :return: (R_A_prime, t_A_prime), A' 座标系在世界座标系下的旋转矩阵和平移向量.
    """
    # 计算 A' 的旋转矩阵
    R_A_prime = R_B_prime @ np.linalg.inv(R_BA)
    
    # 计算 A' 的平移向量
    t_A_prime = t_B_prime - R_A_prime @ t_BA
    R_A_prime=rotation_matrix_to_euler_angles(R_A_prime)
    return R_A_prime, t_A_prime#尤拉角的單位應該為度

def Rotation_matrix(rx,ry,rz):#將歐拉角轉換為旋轉矩陣
    '''
    將歐拉角轉換為旋轉矩陣
    rx,ry,rz為歐拉角

    '''
    # Convert degrees to radians
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)
    
    # Calculate the rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rx), -math.sin(rx)],
                    [0, math.sin(rx), math.cos(rx)]])
    
    R_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                    [0, 1, 0],
                    [-math.sin(ry), 0, math.cos(ry)]])
    
    R_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                    [math.sin(rz), math.cos(rz), 0],
                    [0, 0, 1]])
    
    # Combine the rotation matrices
    R = np.dot(np.dot(R_z, R_y), R_x)
    return R


def euler_to_axis_angle(rx, ry, rz):#欧拉角轉换为轴-角表示(webots內的rotation以轴-角表示)
    """
    Converts Euler angles (in degrees) to axis-angle representation.
    
    Args:
    rx: Rotation around x-axis in degrees.
    ry: Rotation around y-axis in degrees.
    rz: Rotation around z-axis in degrees.
    
    Returns:
    A tuple of four values representing the axis-angle (axis_x, axis_y, axis_z, angle).
    """
    # Convert degrees to radians
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)
    
    # Calculate the rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rx), -math.sin(rx)],
                    [0, math.sin(rx), math.cos(rx)]])
    
    R_y = np.array([[math.cos(ry), 0, math.sin(ry)],
                    [0, 1, 0],
                    [-math.sin(ry), 0, math.cos(ry)]])
    
    R_z = np.array([[math.cos(rz), -math.sin(rz), 0],
                    [math.sin(rz), math.cos(rz), 0],
                    [0, 0, 1]])
    
    # Combine the rotation matrices
    R = np.dot(np.dot(R_z, R_y), R_x)
    
    # Calculate the axis-angle representation
    angle = math.acos((np.trace(R) - 1) / 2)
    sin_angle = math.sin(angle)
    
    if sin_angle > 1e-6:  # Avoid division by zero
        axis_x = (R[2, 1] - R[1, 2]) / (2 * sin_angle)
        axis_y = (R[0, 2] - R[2, 0]) / (2 * sin_angle)
        axis_z = (R[1, 0] - R[0, 1]) / (2 * sin_angle)
    else:
        # If the angle is very small, the axis is not well-defined, return the default axis
        axis_x = 1
        axis_y = 0
        axis_z = 0

    return axis_x, axis_y, axis_z, angle
#------------------------------------------------------工件姿態反推手臂末端軸姿態

def get_transformation_matrix(rotation, translation):
    """
    根據旋轉矩陣和平移向量生成4x4的齊次轉換矩陣
    """
    T = np.eye(4)
    T[:3, :3] = rotation  # 設置旋轉部分
    T[:3, 3] = translation  # 設置平移部分
    return T

def invert_transformation_matrix(T):
    """
    反轉4x4齊次轉換矩陣
    """
    R_inv = T[:3, :3].T  # 旋轉矩陣的轉置
    t_inv = -R_inv @ T[:3, 3]  # 平移向量的反轉
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def quaternion_angle(q1, q2):
    '''
    計算兩個四元數之間旋轉角度 𝜃
    qi = np.array([qx, qy, qz, qw])
    '''
    # 將四元數正規化 (確保它們為單位四元數)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # 計算內積 (dot product)
    dot_product = np.dot(q1, q2)

    # 修正數值誤差，確保內積在 [-1, 1] 範圍內
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # 計算旋轉角度 (θ = 2 * arccos(|q1 ⋅ q2|))
    theta = 2 * np.arccos(abs(dot_product))

    # 轉換為度數並回傳
    return np.degrees(theta)

def euler_to_quaternion(euler_angles, degrees=True):
    """
    將尤拉角轉換成四元數
    :param euler_angles: (rx, ry, rz) 三個旋轉角度 (以弧度或度為單位)
    :param degrees: 是否以度數為單位 (預設為True)
    :return: 四元數 (qx, qy, qz, qw)
    """
    # 使用 SciPy 進行轉換
    r = R.from_euler('xyz', euler_angles, degrees=degrees)
    quaternion = r.as_quat()  # 輸出格式為 [qx, qy, qz, qw]
    return quaternion

def axis_angle_to_quaternion(axis, angle):
    """
    將軸角表示法 (axis, angle) 轉換為四元數。
    :param axis: 旋轉軸 (3D 向量)
    :param angle: 旋轉角度 (弧度)
    :return: 四元數 [x, y, z, w]
    """
    axis = np.array(axis) / np.linalg.norm(axis)  # 確保軸是單位向量
    quaternion = R.from_rotvec(axis * angle).as_quat()
    return quaternion

def quaternion_to_matrix(q):
    """將四元數轉換為旋轉矩陣"""
    return R.from_quat(q).as_matrix()

def matrix_to_quaternion(R_mat):
    """將旋轉矩陣轉換為四元數"""
    return R.from_matrix(R_mat).as_quat()
def compute_target_end_effector_pose(p_w, q_w, p_e, q_e, p_w_new, q_w_new):
        """
        根據工件的初始和目標位姿計算新的末端執行器位姿。
        輸入工件與末端執行器的初始位置以確定兩坐標系之間的相對位置，接著就可以根據工件的目標位置推算出相對應末端執行器的位置
        :param p_w: 初始工件位置 (x, y, z)
        :param q_w: 初始工件四元數 (x, y, z, w)
        :param p_e: 初始末端執行器位置 (x, y, z)
        :param q_e: 初始末端執行器四元數 (x, y, z, w)
        :param p_w_new: 新的工件位置 (x, y, z)
        :param q_w_new: 新的工件四元數 (x, y, z, w)
        :return: (p_e_new, q_e_new) 新的末端執行器位置和四元數
        """
        # 將四元數轉換為旋轉矩陣
        R_w = quaternion_to_matrix(q_w)
        R_e = quaternion_to_matrix(q_e)
        R_w_new = quaternion_to_matrix(q_w_new)
        
        # 轉換為齊次變換矩陣
        T_w = np.eye(4)
        T_w[:3, :3] = R_w
        T_w[:3, 3] = p_w
        
        T_e = np.eye(4)
        T_e[:3, :3] = R_e
        T_e[:3, 3] = p_e
        
        # 計算工件相對於末端執行器的變換矩陣
        T_we = np.linalg.inv(T_w) @ T_e
        
        # 計算新的工件變換矩陣
        T_w_new = np.eye(4)
        T_w_new[:3, :3] = R_w_new
        T_w_new[:3, 3] = p_w_new
        
        # 計算新的末端執行器變換矩陣
        T_e_new = T_w_new @ T_we
        
        # 提取新的位置與旋轉矩陣
        p_e_new = T_e_new[:3, 3]
        R_e_new = T_e_new[:3, :3]
        
        # 轉換為四元數
        q_e_new = matrix_to_quaternion(R_e_new)
        
        return p_e_new, q_e_new

def fix_webots_frame_error(Rotation_matrix):
    # 假設給定 A 座標系相對於世界座標的旋轉矩陣 R
    '''
    由於urdf裡面的末端執行器坐標系與webots裡面的末端執行器坐標系不太一樣,
    先將webots裡面末端執行器的目標角度(以相對世界坐標系的旋轉矩陣表示)轉換成urdf裡面的目標角度,才可以接下來的逆向運動學計算。
    R = np.array([[0, -1, 0], 
                [1,  0, 0], 
                [0,  0, 1]])  # 示例旋轉矩陣
    '''

    # 沿著 A 座標系的 x 軸旋轉 -90 度
    Rx_A = R.from_euler('x', -90, degrees=True).as_matrix()

    # 計算新的旋轉矩陣 R' = R @ Rx_A
    R_prime = Rotation_matrix @ Rx_A
    return R_prime
def get_IK_angle(target_position, target_orientation,initial_position, orientation_axis="all"):
    """
    计算逆向运动学角度
    :param target_position: 目标末端位置 [x, y, z]
    :param target_orientation: 目标方向 (3x3 旋转矩阵)
    :param orientation_axis: 指定对齐轴 ("x", "y", "z")，或者 "all" 进行完整姿态匹配
    :return: 6 轴角度 (弧度制)
    :initial_position: 手臂各軸的初始角度 [a1,a2,a3,a4,a5,a6]
    """
    Initial_Position = np.zeros(10)
    Initial_Position[2:8]=initial_position
    # 计算逆运动学
    ik_angles = robot_chain.inverse_kinematics(
        target_position,
        target_orientation=target_orientation ,
        orientation_mode=orientation_axis,
        initial_position=Initial_Position
    )
    return ik_angles[2:8] # 取 6 轴角度 (去掉基座)

def check_joint_limits(robot_initial_pos):
    complete_robot_initial_pos=[0,0,0,0,0,0,0,0,0,0]
    complete_robot_initial_pos[2:8]=robot_initial_pos
    # print("complete_robot_initial_pos=",complete_robot_initial_pos)
    out_of_limit=False #如果沒超出活動範圍

    for i, link in enumerate(robot_chain.links):
        if link.bounds is not None:
            lower, upper = link.bounds
            angle = complete_robot_initial_pos[i]
            if angle < lower or angle > upper:
                # print(f"[超出範圍] 關節 {i} ({link.name}): "
                #       f"目前角度 {angle:.4f} 超出範圍 ({lower:.4f}, {upper:.4f})")
                out_of_limit=True#超出活動範圍
            # else:
            #     print(f"[正常] 關節 {i} ({link.name}): "
            #           f"{angle:.4f} 在範圍內 ({lower:.4f}, {upper:.4f})")
        # else:
        #     # link 沒有角度限制，可能是固定或虛擬關節
        #     print(f"[無限制] 關節 {i} ({link.name})：不參與IK計算或無限制")
    return out_of_limit

def directly_go_to_target(quaternion,p_w_new,robot_initial_pos):
    """
    輸入工件的目標位置以及目標角度(以軸角表示法表示),以及手臂的初始姿態
    p_w_new:工件目標位置
    axis:旋轉軸
    angle:旋轉角
    robot_initial_pos:手臂各軸的初始角度 [a1,a2,a3,a4,a5,a6]
    quaternion:工件的目標角度(以四位元數表示)
    """
    #輸入工件目標位置的旋轉(軸角表示法)，將其轉換為四位元數
    # axis = [0.654636,0.377894,-0.654712]
    # angle = 2.41868
    #輸入工件目標位置的平移與旋轉()
    # p_w_new = np.array([0.059, 0.646998, 0.46])
    q_w_new = np.array(quaternion)
    #輸入工件與末端執行器初始位置(四位元數表示旋轉)，根據工件目標位置計算末端執行器的目標位置
    p_w = np.array([0.816996, 0.428782, 0.0628227])
    q_w = np.array([ 0.0, -1,  0.0,  1.21326795e-04])  # 單位四元數
    p_e = np.array([0.816992, 0.233936, 0.0628227])
    q_e = np.array([ 0.0, -1,  0.0,  1.21326795e-04])
    p_e_new, q_e_new = compute_target_end_effector_pose(p_w, q_w, p_e, q_e, p_w_new, q_w_new)#末端執行器的目標位置

    R_e_new = quaternion_to_matrix(q_e_new)#將四位元數轉換成旋轉矩陣
    R_e_new=fix_webots_frame_error(R_e_new)#由於urdf的末端執行器坐標系與webots裡面不同，進行修正
    ikAnglesD=get_IK_angle(p_e_new,R_e_new,robot_initial_pos)#利用逆向運動學求出機器手臂各軸角度

    return ikAnglesD


class World(Env):
    def __init__(self):
        """
        action_space 和 observation_space 名稱不能更改，因為這是 gymnasium內部認定的標準屬性
        """
        self.action_space=spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        #---------------------------------------------------------------------------------------------------------------
        # 初始化 Supervisor

        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())

        self.max_steps = 250
        self.current_step = 0
        self.total_step=0
        self.episode_reward = 0.0

        #------------------------------------
        #採樣初始的手臂姿態
        self.saved_joint_angles = [] # 已儲存的姿態
        self.sample_counts = []       # 記錄每個姿態被取樣的次數
        self.init_sample_numbers = [] #紀錄相對應的採樣點號碼
        self.p = 0.9                    # rank-based power factor /  p=0為完全隨機；p=1權重差異明顯
        # init_pose=[1.57,-1.57,1.57,0,1.57,0]
        self.init_pose=[ 1.57,-1.57,1.57,0,1.57,0 ]
        # self.add_pose(init_pose,0)
        self.init_target_sample_num = 0
        #------------------------------------

        # 假設 state_size 是 3 維，action_size 是 1 維的簡單環境
        # self.state_size = 21 #應根據
        # self.action_size = 6 # action_size的數字表示action的動作種類
        #讀取工件目標位置(砂帶接觸點位置)
        #讀取工件目前位置
        #判斷是否發生碰撞

        # self.state = np.random.rand(self.state_size)  # 隨機初始狀態 (這裡只是做初始化，並不需要修改)
        '''
        action_size為6,表示各馬達的角度變化,action[0]為第一個馬達的動作,action[1]為第一個馬達的動作...
        states_size,需要先將所有的states轉為一維向量:[p1,p2,p3,p4,p5,p6,collision,t1,t2,t3,x,y,rz,theta]
        states:
            各馬達位置：[p1,p2,p3,p4,p5,p6]
            是否發生碰撞:collision
            工件目標位置為砂帶接觸點位置：[ct1,ct2,ct3],[cr1,cr2,cr3,ca]
            工件目前位置(position/orientation):[t1,t2,t3],[r1,r2,r3,r4,r5,r6,r7,r8,r9]
            工件目前位置與目標位置之差距：


        '''
        self.grinder=self.supervisor.getFromDef("grinder")
        self.grinder_translation_field= self.grinder.getField("translation")
        #初始化狀態
        #控制手臂各軸馬達
        self.motors = []
        self.motors.append(self.supervisor.getDevice('shoulder_pan_joint'))
        self.motors.append(self.supervisor.getDevice('shoulder_lift_joint'))
        self.motors.append(self.supervisor.getDevice('elbow_joint'))
        self.motors.append(self.supervisor.getDevice('wrist_1_joint'))
        self.motors.append(self.supervisor.getDevice('wrist_2_joint'))
        self.motors.append(self.supervisor.getDevice('wrist_3_joint'))
        for i in range(6):  # UR5e有6個關節
            # joint_name = f"shoulder_lift_joint{i+1}"
            self.motors[i].setPosition(0)  # 初始位置設為0
        # pose_node = supervisor.getFromDef("end_effector")
        # target_position=[0.6,0,0.4]
        # orientation=[1,0,0,0]
        self.sensors = []
        for motor in self.motors:
            sensor = motor.getPositionSensor()
            sensor.enable(self.timestep)
            self.sensors.append(sensor)

       #----------------------------------------------------
        '''
        讀取路徑資訊之後，只取路徑上最後一採樣點(做為每個episode的起點)以及下一條路徑第一個採樣點(做為每個episode的目標),
        根據episode的起點位置的工件姿態用逆向運動學
        '''
        file_path = "simple_2_7cm_3505_only4faces.csv"
        df = pd.read_csv(file_path, header=None)

        # 建立一個字典來存儲 (面編號, 路徑編號) 對應的所有 index
        self.face_path_groups = defaultdict(list)
        for idx in range(len(df)):
            face_id = df.iloc[idx, 0]  # 第幾個面
            path_id = df.iloc[idx, 1]  # 第幾條路徑
            self.face_path_groups[(face_id, path_id)].append(idx)
        for key, indexes in self.face_path_groups.items():
            print(f"面 {key[0]}, 路徑 {key[1]} 的索引有: {indexes}")

        # 建立 index -> (face_id, path_id) 的映射
        self.index_to_group = {}
        for (face_id, path_id), indexes in self.face_path_groups.items():
            for idx in indexes:
                self.index_to_group[idx] = (face_id, path_id)


        #砂帶接觸點(世界坐標系底下)的旋轉與平移
        R_contactpoint_frame = Rotation_matrix(-90,-90,0)#角度單為為度,此處為砂帶上接觸點的座標的旋轉,輸入歐拉角rx,ry,rz轉換成旋轉矩陣
        t_contactpoint_frame = np.array([0.0100017, 0.5567, 0.41])#此處為砂帶上接觸點的座標，可由"座標轉換.ipynb"計算其歐拉角(軸-角--->歐拉)

        '''
        讀取路徑採樣點資訊後，採樣點位置為相對於工件坐標系之坐標系，將採樣點的坐標系(在工件坐標系底下)的旋轉與平移，以及砂帶接觸點(世界坐標系底下)的旋轉與平移作為輸入，
        輸出工件坐標系在世界坐標系底下的平移(t_A_prime)與旋轉(R_A_prime)
        calculate_A_prime:輸入工件路徑採樣點研磨時的座標平移與歐拉角，輸出工件座標研磨時的平移與歐拉角
        '''
        self.samplepoint_num=0 #設定目前是在第幾個採樣點
        self.total_samplepoint_num=len(df)
        print("total points=",self.total_samplepoint_num)
        self.t_toolframes=[]
        self.r_toolframes=[]
        self.R_A_primes=[]
        self.t_A_primes=[]
        last_point_on_trajectory_indexes=[]
        self.samplepoint_num=0 #設定目前的目標是要前往第幾個採樣點

        # 比較的是第二列 (索引為 1) 的數字,也就是這個加工面的第幾條路徑
        column_to_compare = 1

        for index in range(self.total_samplepoint_num):  # 遍歷所有行，最後一行無法與下一行比較

            current_value = df.iloc[index, column_to_compare]  # current_value當前採樣點(起點)是第幾條路徑
            # print("index=",index)
            if index!=self.total_samplepoint_num-1:
                next_value = df.iloc[index + 1, column_to_compare]  # 下一行的值(目標點)
            else:
                next_value=0
            
            samplepoint_info=df.iloc[index].values
            #-------------------------------------------------------------------------------------------------
            R_samplepoint=Rotation_matrix(samplepoint_info[5],samplepoint_info[6],samplepoint_info[7])
            t_samplepoint = np.array([samplepoint_info[2]/1000,samplepoint_info[3]/1000,samplepoint_info[4]/1000])
            # B' 座标系在世界座标系下的旋转矩阵和平移向量-->此處B'應設定為砂帶上的接觸點座標系

            # 计算 A' 在世界座标系下的表示-->研磨過程中工件座標系在世界座標系的位置
            R_A_prime, t_A_prime = calculate_A_prime(R_samplepoint, t_samplepoint, R_contactpoint_frame, t_contactpoint_frame)#工件坐標系相對世界座標系的旋轉(已尤拉角表示)與平移
            self.R_A_primes.append(R_A_prime)
            self.t_A_primes.append(t_A_prime)
            t_toolframe=[t_A_prime[0],t_A_prime[1],t_A_prime[2]]#工件座標系相對於世界坐標系的平移
            # r_toolframe=euler_to_axis_angle(R_A_prime[0],R_A_prime[1],R_A_prime[2])#工件座標系相對於世界坐標系的旋轉(以軸角表示法表示)
            r_toolframe=euler_to_quaternion(R_A_prime)#工件座標系相對於世界坐標系的旋轉(以四位元數表示法表示)
            self.r_toolframes.append(r_toolframe)
            self.t_toolframes.append(t_toolframe)


            
            #------------------------------------------------------------------------------------------------------
            if current_value != next_value:  # 若值不同,表示該採樣點為該路徑上最後一個採樣點,記錄下index
                last_point_on_trajectory_indexes.append(index)
                # print("index=",index)
                # print(f"Row {index}: Value = {current_value}")
                # print("Row data:", df.iloc[index].values)  # 打印第 n 行的完整資訊
                # print("Row data[2]:", df.iloc[index].values[2])
                # print(f"Row {index + 1}: Value = {next_value}")
                # print("Row data (n+1):", df.iloc[index + 1].values)  # 打印第 n+1 行的完整資訊

        # """
        # 一次只訓練一個面
        # face1[0:55]
        # face2[0:118]
        # face3[118:173]
        # face4[173:215]
        # face5[215:248]
        # face6[248:311]
        # face7[311:]
        # """
        # self.r_toolframes = self.r_toolframes[0:55]
        # self.t_toolframes = self.t_toolframes[0:55]

        self.get_init_point_on_path()
        next_target_point=self.random_choose_a_target_point()
        print(next_target_point)
        #-------------------------------------------------------------------------------------------------------

        for i, link in enumerate(robot_chain.links):
            print(f"Link {i}: {link.name}")
            if link.bounds is not None:
                print(f"  Bounds: {link.bounds}")
            else:
                print("  This link is fixed or passive (no bounds)")
        #-------------------------------------------------------------------------------------------------------
    def check_if_its_in_the_same_path(self,index1 ,index2):
        # 判斷是否屬於同一個面與路徑
        if self.index_to_group.get(index1) == self.index_to_group.get(index2):
            # print(f"索引 {index1} 與 {index2} 是同一個面與同一條路徑")
            return True
        else:
            # print(f"索引 {index1} 與 {index2} 不在同一個面或同一條路徑")
            return False
        
    def get_init_point_on_path(self):
        '''
        採樣出每條路徑的起始點位置
        只需要回傳該採樣點的號碼(self.samplepoint_num)
        '''
        self.start_points=[]
        self.end_points=[]

        self.start_points.append(0)
        self.samplepoint_num=0 #設定目前是在第幾個採樣點
        for i in range(self.total_samplepoint_num-1):
            if self.check_if_its_in_the_same_path(i,i+1)==False:
                self.start_points.append(i+1)
                self.end_points.append(i)
        self.end_points.append(self.total_samplepoint_num-1)

        print("start points = ",self.start_points)
        print("end points = ",self.end_points)

        return self.start_points
    
    def random_choose_a_target_point(self,already_chosen_points = []):
        available_choices = [x for x in self.start_points if x not in already_chosen_points]

        if available_choices:
            next_target_point=random.choice(available_choices)
            return next_target_point
        
        else:
            print("已完成所有點位，沒有可選的點位了！")
            self.success = True #完成所有點位，記錄成功率
            return None

        
    def get_motor_angles(self):
        #各馬達上加入sensor測量轉角位置
        # sensors = []
        # for motor in self.motors:
        #     sensor = motor.getPositionSensor()
        #     # sensor.enable(timestep)
        #     sensors.append(sensor)

        joint_angles = [sensor.getValue() for sensor in self.sensors]#讀取各軸位置

        return (np.array(joint_angles))
    
    def get_state(self):
        '''
        獲取當前的狀態
        states:
            各馬達位置：[p1,p2,p3,p4,p5,p6]
            是否發生碰撞:collision
            工件目標位置(研磨路徑採樣點基於砂帶接觸點位置推算至工件坐標系):[ct1,ct2,ct3],[cr1,cr2,cr3,ca]
            工件目前位置(position/orientation):[t1,t2,t3],[r1,r2,r3,a]
            工件目前位置與目標位置之差距：
        '''
        # 返回當前狀態

        #馬達位置
        joint_angles=self.get_motor_angles()
        '''
        self.samplepoint_num:目前在第幾個採樣點
        '''
        # print(f"目標是到達第{self.samplepoint_num}個採樣點")
        t_toolframe=[self.t_toolframes[int(self.samplepoint_num)]][0]#工件目標位置的平移
        # print("target position=",t_toolframe)
        r_toolframe=[self.r_toolframes[int(self.samplepoint_num)]]
        r_toolframe=r_toolframe[0]
        r_toolframe=[r_toolframe[0],r_toolframe[1],r_toolframe[2],r_toolframe[3]] #工件目標位置的旋轉(軸角法表示)
        # print("target orientation=",r_toolframe)
        #工件目前位置
        solid_node = self.supervisor.getFromDef("simple_2_show")#獲得工件模型
        if solid_node is None:
            raise ValueError("Solid node not found")
        #----------------------------------------------------------------
        '''
        工件的相對平移與旋轉
        '''
        # workpiece_translation_field=solid_node.getField('translation')
        # workpiece_rotation_field=solid_node.getField('rotation')
        # workpiece_current_translation = workpiece_translation_field.getSFVec3f()#工件目前位置的平移
        # workpiece_current_rotation = workpiece_rotation_field.getSFVec3f()#工件目前位置的旋轉(軸角法表示)
        #-------------------------------------------------------------------
        '''
        工件的在世界坐標系下的平移與旋轉
        '''
        workpiece_current_translation = solid_node.getPosition()
        # print("workpiece current position:", workpiece_current_translation)
        rotation = solid_node.getOrientation()

        #轉換為四元數或歐拉角
        rotation_matrix = np.array(rotation).reshape(3, 3)
        # #轉換為四元數
        workpiece_current_rotation = R.from_matrix(rotation_matrix).as_quat()
        # print("current orientation:", workpiece_current_rotation )

        #-------------------------------------------------------------------

        #是否發生碰撞
        '''
        讀取各軸solid的資訊,得到接觸點數量,接觸點數量大於0表示手臂與環技發生碰撞==>collision=1
        '''

        shoulder_link_solid_node = self.supervisor.getFromDef("shoulder_link_solid")#shoulder_link_solid/upper_arm_link_solid/forearm_link_solid/wrist_1_link_solid
        contact_points_1 = shoulder_link_solid_node.getContactPoints()
        upper_arm_link_solid_node = self.supervisor.getFromDef("upper_arm_link_solid")#shoulder_link_solid/upper_arm_link_solid/forearm_link_solid/wrist_1_link_solid
        contact_points_2 = upper_arm_link_solid_node.getContactPoints()
        forearm_link_solid_node = self.supervisor.getFromDef("forearm_link_solid")#shoulder_link_solid/upper_arm_link_solid/forearm_link_solid/wrist_1_link_solid
        contact_points_3 = forearm_link_solid_node.getContactPoints()
        wrist_1_link_solid_node = self.supervisor.getFromDef("wrist_1_link_solid")#shoulder_link_solid/upper_arm_link_solid/forearm_link_solid/wrist_1_link_solid
        contact_points_4 = wrist_1_link_solid_node.getContactPoints()
        wrist_2_link_solid_node = self.supervisor.getFromDef("wrist_2_link_solid")#shoulder_link_solid/upper_arm_link_solid/forearm_link_solid/wrist_1_link_solid
        contact_points_5 = wrist_2_link_solid_node.getContactPoints()
        wrist_3_link_solid_node = self.supervisor.getFromDef("wrist_3_link_solid")#shoulder_link_solid/upper_arm_link_solid/forearm_link_solid/wrist_1_link_solid
        contact_points_6 = wrist_3_link_solid_node.getContactPoints()

        workpiece_solid_node = self.supervisor.getFromDef("simple_2_show")#shoulder_link_solid/upper_arm_link_solid/forearm_link_solid/wrist_1_link_solid
        contact_points_7 = workpiece_solid_node.getContactPoints()
        if len(contact_points_1) > 0 or len(contact_points_2) > 0 or len(contact_points_3) > 0 or len(contact_points_4) > 0 or len(contact_points_5) > 0 or len(contact_points_6) > 0 :
            
            # print("撞到啦啊啊啊啊啊啊啊啊啊")
            # print("len(contact_points_1)=",len(contact_points_1))
            # print("len(contact_points_2)=",len(contact_points_2))
            # print("len(contact_points_3)=",len(contact_points_3))
            # print("len(contact_points_4)=",len(contact_points_4))
            # print("len(contact_points_5)=",len(contact_points_5))
            # print("len(contact_points_6)=",len(contact_points_6))
            collision=1
            
        elif len(contact_points_7)>0:
            """
            允許工件稍微接觸
            """
            collision = 0.1
        else:

            collision=0
        # print("joint_angles[0]=",joint_angles[0],collision, t_toolframe[0])
        state=np.array([joint_angles[0],joint_angles[1],joint_angles[2],joint_angles[3],joint_angles[4],joint_angles[5],collision,
                       t_toolframe[0],t_toolframe[1],t_toolframe[2],r_toolframe[0],r_toolframe[1],r_toolframe[2],r_toolframe[3],
                       workpiece_current_translation[0],workpiece_current_translation[1],workpiece_current_translation[2],
                       workpiece_current_rotation[0],workpiece_current_rotation[1],workpiece_current_rotation[2],workpiece_current_rotation[3]])
        """
        : joint angles:state[0:6]
        : collision:state[6]
        : target workpiece position:state[7:10]
        : target workpiece orientation:state[10:14]
        : current workpiece position:state[14:17]
        : current workpiece orientation:state[17:]
        """



        # return torch.from_numpy(state).float()
        return state
    def distance_btw_target_angles(self,motor_init_position):#計算手臂目前姿態與目標姿態之間的差距
        # motor_init_position=np.array([0,0,0,0,0,0])#目前設定手臂的起始位置各馬達角度位置為[0,0,0,0,0,0],應改為每個episode的起始姿態
        
        current_angle=self.get_motor_angles()
        delta_angle=motor_init_position-current_angle
        delta=(delta_angle[0]**2+delta_angle[1]**2+delta_angle[2]**2+delta_angle[3]**2+delta_angle[4]**2+delta_angle[5]**2)**0.5
        return delta
    

    #---------------------------------------------------------------------------------------------------------------------------

    def reset(self, seed=None, options=None):#重置世界
        
        super().reset(seed=seed) 
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()

        while self.supervisor.step(self.timestep) != -1:
            self.grinder_translation_field.setSFVec3f([0,0,100])#先將沙袋機一首避免撞到
            break
        # joints_angle=[1.57,-1.57,1.57,0,1.57,0]
        delta=1
        t=0
        while delta >= 0.0005 and self.supervisor.step(self.timestep) != 1:
            for n, motor in enumerate(self.motors):
                motor.setPosition(self.init_pose[n])    
            current_angles=self.get_motor_angles() 
            delta = np.linalg.norm(np.array(self.init_pose) - np.array(current_angles))
            t=t+1
            if t>=100:
                break

        while self.supervisor.step(self.timestep) != -1:
            self.grinder_translation_field.setSFVec3f([0.3,0.95,1.03529e-08])#手臂移動到初始位置後再將砂帶機台拿回來
            break

        # reset parameters 
        self.already_chosen_points=[]
        self.success = False       
        self.done = False 
        self.get_to_target_times=0
        self.current_step = 0
        self.episode_reward = 0.0
        self.samplepoint_num = self.random_choose_a_target_point()
        self.min_pos_error=1
        self.min_ori_error=180
        self.min_error=[self.min_pos_error,self.min_ori_error]

        self.quaternion_history = deque(maxlen=11)  # 記錄過去 11 步的 工件四位元數


        
        info = {}
        self.supervisor.step(self.timestep)
        # print("#######################  Reset!!  ################################")
        # print("success_rate = ",tracker.get_success_rate())
        # print("initail target point:",self.samplepoint_num)
        # face_id = self.index_to_group[self.samplepoint_num][0]
        # print("target face = ",face_id)

        
        observation = self.get_state()

        q1_current=np.array([observation[10],observation[11],observation[12],observation[13]])
        q2_current=np.array([observation[17],observation[18],observation[19],observation[20]])
        self.init_orientation_error=quaternion_angle(q1_current,q2_current)

        return observation, info

    def move_to_target_with_ikpy(self,quaternion,p_w_new,robot_initial_pos):
        """
        當工件位置及角度達到目標範圍內,直接利用ikpy控制手臂移動至精確的目標位置
        """

        joints_angle=directly_go_to_target(quaternion,p_w_new,robot_initial_pos)

        delta=1
        t=0
        while delta >= 0.0005 and self.supervisor.step(self.timestep) != 1:
            # for i in range(6):  # UR5e有6個關節
            #     # joint_name = f"shoulder_lift_joint{i+1}"
            #     self.motors[i].setPosition(0)#這裡應該要修改成目前的起始位置(比如第n條路徑的最後一個採樣點)
            for n, motor in enumerate(self.motors):
                motor.setPosition(joints_angle[n])    
            current_angles=self.get_motor_angles() 
            delta = np.linalg.norm(np.array(joints_angle) - np.array(current_angles))
            t=t+1
            if t>=100:
                break


    
    def calculate_reward(self,current_state,next_state):
        '''
        獎勵函數:reward = r1 + r2 + r3 + r4
        r1=與目標姿態的平移差距
        r2=與目標姿態的旋轉差距
        r3=達成目標獎勵(當目前工件姿態與目標工件姿態之平移與旋轉，差距皆小於閥值)
        r4=碰撞懲罰(發生碰撞collision=1)
        累積作動量(不確定要不要加?)
        平滑度(不確定要不要加?)
        所花費步數(不確定要不要加?)
        '''
        threshold_1=0.001#利用強化學習控制
        threshold_2=1
        a1= 15
        a2= 5
        a3=1
        a4=1.5
        a5 = 0.01
        a6 = 10
        gama=1
        #平移誤差(以勢能表示)------------------------------------------------------------------------
        position_error=((current_state[7]-current_state[14])**2+(current_state[8]-current_state[15])**2+(current_state[9]-current_state[16])**2)**0.5
        # print("position_error=",position_error)
        position_error_prime=((next_state[7]-next_state[14])**2+(next_state[8]-next_state[15])**2+(next_state[9]-next_state[16])**2)**0.5
        # print("position_error_prime=",position_error_prime)
        r1=-(gama*position_error_prime-position_error)
        #角度誤差(以勢能表示)------------------------------------------------------------------------
        q1_current=np.array([current_state[10],current_state[11],current_state[12],current_state[13]])
        q2_current=np.array([current_state[17],current_state[18],current_state[19],current_state[20]])
        orientation_error=quaternion_angle(q1_current,q2_current)
        # print("q1_current=",q1_current)
        # print("q2_current=",q2_current)
        # print("--------------------------------------")
        # print("orientation_error=",orientation_error)

        q1_next=np.array([next_state[10],next_state[11],next_state[12],next_state[13]])
        q2_next=np.array([next_state[17],next_state[18],next_state[19],next_state[20]])
        orientation_error_prime=quaternion_angle(q1_next,q2_next)
        r2=-(gama*orientation_error_prime-orientation_error)/self.init_orientation_error
        # print("q1_next=",q1_next)
        # print("q2_next=",q2_next)
        # print("orientation_error_prime=",orientation_error_prime)

        #誤差倒數為獎勵--------------------------------------------------------------------
        if position_error_prime>=threshold_1:#平移誤差
            r3=threshold_1/position_error_prime
        else:
            r3=1

        # r4=-r3*orientation_error_prime/self.init_orientation_error
        
        """
        此獎勵函數的設計是為了滿足以下兩點:
        1. r4需要小於0
        2. 角度誤差越小，獎勵增加幅度要越大
        """
        if orientation_error_prime>=threshold_2:#旋轉誤差
            r4=threshold_2/orientation_error_prime
        else:
            r4=1
        # # r2=r3*r2
        r4=r3*(r4-1) 
        # r3=r4*(r3-1)


        #碰撞懲罰------------------------------------------------------------------------
        r5=-next_state[6]
        # if next_state[6]!=1:#如果手臂本身沒發生碰撞(只有工件碰撞or無碰撞)，碰撞懲罰隨著越接近目標獎勵而降低
        #     # total_error=position_error_prime/0.05+ orientation_error_prime/30
        #     total_error=orientation_error_prime/30
        #     if position_error_prime>0.05 or orientation_error_prime>30:
        #         r5=-next_state[6]
        #     else:
        #         r5=total_error*r5
        # if position_error_prime>0.08 and orientation_error_prime>40 and r5 == -0.1: 
        #     r5 = 0

        #------------------------------------------------------------------------------
        if (position_error_prime<0.01 and orientation_error_prime<10) :
            r6 = 1
            get_to_target=True
        else:
            r6=0
            get_to_target=False
        if next_state[6]==1 or next_state[6]==0.1:
            crash = True
        else:
            crash = False
        #------------------------------------------------------------------------------  
        #應該要改為：五步之後角度還維持在同樣的位置，表示卡住了
        self.quaternion_history.append(q2_current)
        r7=0
        if len(self.quaternion_history) == 11:  # 有足夠長度可比較
            angle_variation=quaternion_angle(self.quaternion_history[-1],self.quaternion_history[0])
            # print("angle_variation=",angle_variation)
            if abs(angle_variation) < 2 and not get_to_target:
                r7 = -0.001
            else:
                r7 = 0

        #各軸角度變化量-------------------------------------------------------------------
        delta_angle_1 = abs(current_state[0] - next_state[0])
        delta_angle_2 = abs(current_state[1] - next_state[1])
        delta_angle_3 = abs(current_state[2] - next_state[2])
        delta_angle_4 = abs(current_state[3] - next_state[3])
        delta_angle_5 = abs(current_state[4] - next_state[4])
        delta_angle_6 = abs(current_state[5] - next_state[5])
        r8 = -(0.2*delta_angle_2+0.1*delta_angle_3+0.1*delta_angle_4 + 0.1*delta_angle_5+0.1*delta_angle_6)/1000

        #------------------------------------------------------------------------------
        # print("r1=",r1)
        # print("r2=",r2)
        # print("r3=",r3)
        # print("r4=",r4)
        # print("r5=",r5)
        reward = a1*r1+a2*r2+a3*r3+a4*r4+a5*r5+a6*r6 
        rewards=[r1,r2,r3,r4,r5,r6,r7,r8]
        errors=[position_error_prime,orientation_error_prime]
        return float(reward),get_to_target,crash,rewards,errors
    
    def print_memory(self):
        process = psutil.Process(os.getpid())
        mem_MB = process.memory_info().rss / 1024 / 1024
        return mem_MB

    def step(self, action):
        # print(f"-----------------timestep  {self.current_step}--------------------------------------------------------------------")
        # print(f"目標是到達第{self.samplepoint_num}個採樣點")
        self.current_step += 1
        self.total_step += 1
        '''
        # 執行動作並返回下一個狀態、獎勵和是否完成
        # 如果任務完成,更新目標點位置(self.samplepoint_num加一)
        action=[d_a1,d_a2,d_a3,d_a4,d_a5,d_a6],其中d_a為-1到1之間的值,表示個軸馬達角度變化
        joint_angles=(joint_angles+action*5*2*np.pi/360):
        joint_angles為目前各軸馬達目前角度位置
        action*5*2*np.pi/360為角度變化,在正負5度之間
        '''
        current_state=self.get_state()
        # print("target position=",current_state[7:10])
        # print("target orientation=",current_state[10:14])
        # print("workpiece current position:", current_state[14:17])
        # print("current orientation:", current_state[17:] )
        #-----------------------------------------------------------------------------------------------------
        '''
        這部分是使用馬達[位置]控制機器手臂
        '''
        joint_angles=self.get_motor_angles()#讀取各軸馬達目前角度位置
        target_joint_angles=(joint_angles+action*2*np.pi/360)#將當前各馬達位置加上角度變化量，表示馬達將要移動到的位置(以弧度表示)

        # print("joint_angles 1213=",joint_angles)

        t=0
        while np.all(np.round(target_joint_angles, 2) == np.round(joint_angles, 2)) == False and self.supervisor.step(self.timestep) != 1:
            for i, motor in enumerate(self.motors):
                motor.setPosition(target_joint_angles[i])#執行各馬達動作

            t=t+1
            if t>10:
                break#計數器 t 及 if t > 4: break 的設計主要是為了避免無窮迴圈(如果馬達無法到達目標角度（例如因為碰撞、馬達受限或模擬異常)使程式卡住），確保程式能夠適當地退出 while 迴圈)
            joint_angles=self.get_motor_angles()
        #--------------------------------------------------------------------------------------------------------------------
        '''
        這部分是使用馬達[轉速]控制機器手臂
        '''
        # print("motor velocity=",action)
        # for i, motor in enumerate(self.motors):
        #     motor.setPosition(float('inf'))
        #     motor.setVelocity(action[i])#直接用馬達轉速控制手臂

        
        #--------------------------------------------------------------------------------------------------------------------
        

        
        #執行完action之後會得到的state #應改為從webots內獲得執行動作後的狀態(各軸角度、目標姿態、是否發生碰撞、與目標姿態差異(平移/旋轉))
        next_state=self.get_state()

 
        reward,get_to_target,crash,rewards,errors=self.calculate_reward(current_state,next_state)#根據next_state決定獎勵,達到目標前done都等於0
        if get_to_target:
            self.done = True
            self.already_chosen_points.append(self.samplepoint_num)
            self.current_step=0#達到目標位姿，重置步數
            self.get_to_target_times=self.get_to_target_times+1#紀錄單一episode內成功達到目標點位的次數
            
            # self.samplepoint_num=self.samplepoint_num+1 #從calculate_reward得到get_to_target=True,表示已達到目標點,應更新下一個目標點位(先記錄下當下的手臂姿態，可做為下個episode的初始姿態)
            # if self.samplepoint_num==self.total_samplepoint_num-1: #若最後一個採樣點已經完成,回到第一個點
            #     self.samplepoint_num=0
            '''
            如果get_to_targete==1,表示工件的與目標位姿之間的差異已經小於閥值閥值
            '''
            robot_initial_pos=self.get_motor_angles()
            workpiece_target_quaternion=next_state[10:14]
            workpiece_target_position=next_state[7:10]
            self.move_to_target_with_ikpy(workpiece_target_quaternion,workpiece_target_position,robot_initial_pos)

            #如果下個採樣點和目前的採樣點在同一條路徑上，直接利用ikpy移動工件
            while self.check_if_its_in_the_same_path(self.samplepoint_num,self.samplepoint_num+1):
                self.samplepoint_num=self.samplepoint_num+1
                next_state=self.get_state()
                
                if next_state[6]==1:
                    reward=-1
                    crash=True
                    break

                robot_initial_pos=self.get_motor_angles()
                workpiece_target_quaternion=next_state[10:14]
                workpiece_target_position=next_state[7:10]
                if check_joint_limits(robot_initial_pos):#超出機器手臂的活動範圍
                    crash=True
                    break
                self.move_to_target_with_ikpy(workpiece_target_quaternion,workpiece_target_position,robot_initial_pos)
            
            #隨機選擇下一個目標點位
            # self.samplepoint_num=self.random_choose_a_target_point(self.already_chosen_points)#隨機選擇一個目標點位作為下一個目標(排除已經達成的點位)
            # if self.samplepoint_num == None:#如果所有點位都已經完成，結束這個
            #     self.done = True




        #設定終止條件
        #如果大於最大步數，truncated = True，重置整個episode
        if self.current_step > self.max_steps:
            truncated = True
            tracker.record(self.success)
        else:
            truncated = False
        #如果完成最後一個採樣點，或是發生碰撞，done=True，重置整個episode
        if (len(self.already_chosen_points) == len(self.start_points)) or crash:
            
            self.done = True
            tracker.record(self.success)


        self.episode_reward += reward

        if errors[0]<=self.min_pos_error:
            self.min_pos_error=errors[0]
        if errors[1]<=self.min_ori_error:
            self.min_ori_error=errors[1]
        self.min_error=[self.min_pos_error,self.min_ori_error]



        """
        done=True:	表示 這個 episode 結束了，不管是成功還是失敗
        truncated=True:	表示 episode 是因為 時間上限或步數限制 被「截斷」結束的

        done=True + truncated=False	: 正常結束（成功達成任務或失敗）
        done=True + truncated=True	: 任務結束（成功/失敗）且時間到（較少見
        done=False + truncated=True : 被時間限制中止
        done=False + truncated=False : 任務還在進行中
        """


        info = {
            "timestep reward": reward,
            "timestep rewards": rewards,
            "get to targat":get_to_target
            
        }
        #---------------------------------
        #每一萬步紀錄記憶體使用量
        # if self.total_step % 10000 == 0:
        #     mem = self.print_memory()
        #     info["memory used"]=mem
        #----------------------------------




        if self.done or truncated:
            info["episode reward"] = self.episode_reward
            info["sample point number"] = self.samplepoint_num
            info["success rate"] = tracker.get_success_rate()
            info["get_to_target_times"] = self.get_to_target_times
            # info["min errors"] = self.min_error
            if self.samplepoint_num != None:
                face_id = self.index_to_group[self.samplepoint_num][0]
                info[f"min pos errors/face_{face_id}"] = self.min_pos_error
                info[f"min orien errors/face_{face_id}"] = self.min_ori_error
            
        
        """
        info = {
            "timestep rewards": [r1, r2, r3, r4, r5 ,r6],
            "errors": [error1, error2]
        }
        print(info["timestep rewards"])  # 印出 [r1, r2, r3]
        print(info["errors"][0])  # 印出 error1
        """


        return next_state, reward, self.done, truncated, info
    
#---------------------------------------------------------------------------------------
    def select_samplepoint_num(self, samplepoint_num):
        self.samplepoint_num = samplepoint_num
    
#-------------------------------------------------------------------------------------主循環
# supervisor.step(int(supervisor.getBasicTimeStep()))






