from webots_grinding_env import World
from controller import Robot, Supervisor,Connector
from ikpy.chain import Chain
from gym.spaces import Box
import tensorflow as tf

import torch
torch.autograd.set_detect_anomaly(True)
import numpy as np
import torch
import datetime

# for training
from stable_baselines3 import PPO   
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import CallbackList
import numpy as np
from controller import Robot, Supervisor,Connector
from ikpy.chain import Chain
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

"""
這部分程式碼用於引用訓練好的路徑串接模型，執行路徑串接的動作

程式碼的功能包含:
1.計算並紀錄路徑總長
2.給定指定的目標採樣點，工件由當前位置移動至目標位姿
"""

world = World()
timestep = int(world.timestep)# 設定時鐘（time step）

# # 載入之前訓練好的模型
env = DummyVecEnv([lambda: world])          # SB3 需要 VecEnv 格式

# # 載入已訓練的 PPO 模型
model_path = "lowlevel_models/20250614-203811_4faceS_wp7cm_crash_keeptrain_successrate_1433600_steps"  # <-- 替換成你自己的模型路徑
# model = PPO.load(model_path, env=env, device="cuda")  # 如果有用 GPU 的話



class path_connect_env(Env):
    def __init__(self):
        self.model = PPO.load(model_path, env=env, device="cuda")

    def distance_between_points(self,point,next_point):
        distance=((point[0]-next_point[0])**2+(point[1]-next_point[1])**2+(point[2]-next_point[2])**2)**0.5
        return distance
    
    def add_noise_to_pose_matrix(self,pose_matrix, noise_level=0.015):
        pose_array = np.array(pose_matrix)  # shape (1, 21)
        noise = np.random.uniform(-noise_level, noise_level, size=pose_array.shape)
        return (pose_array + noise).tolist()
    
    def execute_task(self,samplepoint_num):
        """
        輸入採樣點index,透過低層模型執行路徑串接的動作
        """
        # env.set_current_path(target_path_id)  # 設定目前要研磨的路徑
        world.select_samplepoint_num(samplepoint_num)#執行動作
        # print("sample num =",samplepoint_num)
        obs = world.get_state()
        obs = obs.reshape((1,21))
        obs=self.add_noise_to_pose_matrix(obs)
        done = False
        total_displacement=0
        steps=0

        while not done:
            workpiece_current_position = obs[0][14:17]
            action, _ = self.model.predict(obs)
            obs, reward, truncated, done, info = world.step(action[0])
            obs = obs.reshape((1,21))
            workpiece_next_position = obs[0][14:17]
            displacement = self.distance_between_points(workpiece_current_position,workpiece_next_position)

            total_displacement=total_displacement+displacement
            steps=steps+1

            if "get to targat" in info:
                if info["get to targat"]:
                    done=True
            # total_reward += reward
        return total_displacement,steps, info #這部分應該要計算好路徑總長相關的reward，再回傳到主程式



class path_select_env(Env):
    def __init__(self):
        """
        action_space 和 observation_space 名稱不能更改，因為這是 gymnasium內部認定的標準屬性
        """
        super().__init__()
        self.taget_start_points = world.start_points
        self.used_indices=[]
        

        self.worker = path_connect_env()
        self.total_path_length = 0
        n_paths=len(self.taget_start_points)
        print(f"總共有{n_paths}條路徑")

        # 高層動作空間：選擇一條路徑 ID
        self.action_space = spaces.Discrete(n_paths)

        # 高層狀態空間：完成狀態（可以擴充）
        # self.observation_space = spaces.MultiBinary(num_paths)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_paths+7,), dtype=np.float32)

    # def distance_between_points(self,point,next_point):
    #     distance=((point[0]-next_point[0])**2+(point[1]-next_point[1])**2+(point[2]-next_point[2])**2)**0.5
    #     return distance

    # def path_connect(self):
    #     # 開始執行測試 Episode
    #     done = False
    #     episode_reward = 0
    #     step_count = 0
    #     obs=world.get_state()
    #     while not done:
    #         workpiece_current_position = obs[14:17]
    #         action, _ = model.predict(obs, deterministic=True)  # ✅ 採用最優策略
    #         obs, reward, done, info = env.step(action)
    #         workpiece_next_position = obs[14:17]
    #         displacement = self.distance_between_points(self,workpiece_current_position,workpiece_next_position)
    #         episode_reward += reward[0]  # reward 是 list（VecEnv 包裝）
    #         self.displacement_sum += displacement#目前所行進的總路徑長
    #         step_count += 1

    def reset(self, seed=None, options=None):#重置世界
        super().reset(seed=seed) 
        # self.supervisor.simulationReset()
        # self.supervisor.simulationResetPhysics()
        world.reset()
        observation = self.get_state()
        info = {}
        self.total_path_length = 0
        self.total_steps = 0
        self.finish_path_num = 0
        self.used_indices=[]
        self.done_paths = []
        self.done = False
        self.episode_reward = 0
        # print("reselt!!")
        # print("------------------------------")
        return observation, info
    
    def calculate_reward(self,repeat_selection):
        a1=1
        # a2=1/500
        a2=1
        a3=0
        if repeat_selection:
            r1=-1
        else:
            r1=0

        # if set(self.taget_start_points).issubset(set(self.done_paths)):#已完成的路徑中已包含了所有目標路徑==>所有目標點都至少執行過一次
        #     #表示所有目標點都至少執行過一次
        #     r2 = -self.total_steps
        # else:
        #     r2 = 0

        if set(self.taget_start_points).issubset(set(self.done_paths)):#已完成的路徑中已包含了所有目標路徑==>所有目標點都至少執行過一次
            #表示所有目標點都至少執行過一次
            r2 = -self.total_path_length
        else:
            r2 = 0         
        r3 = self.finish_path_num

        reward=a1*r1+a2*r2+a3*r3

        return float(reward)
    
    def get_state(self):
        """
        : current workpiece position:world_state[14:17]
        : current workpiece orientation:world_state[17:]
        """
        world_state = world.get_state()
        current_workpiece_position = world_state[14:17]
        current_workpiece_orientation = world_state[17:]

        obs_path_mask = np.zeros(len(self.taget_start_points))
        obs_path_mask[self.used_indices] = 1.0
        # state=[self.total_path_length,]
        
        obs = np.concatenate([
            current_workpiece_position,
            current_workpiece_orientation,
            obs_path_mask
        ]).astype(np.float32)

        return obs
    
    def step(self, action):
        target_point_index=self.taget_start_points[action]
        # print("step start-----------------------------")

        if target_point_index not in self.done_paths:
            repeat_selection = False

        else:
            repeat_selection = True
            # print("repeat sampling QQ!!")
            # self.done = True
        
        displacement,steps,info= self.worker.execute_task(target_point_index) #選擇完目標採樣點後，進行路徑串接
        if repeat_selection == False: #如果沒有重複選擇路徑，已完成路徑數量+1
            self.finish_path_num = self.finish_path_num + 1
        # print("world.samplepoint_num = ",world.samplepoint_num)

        self.used_indices.append(action)
        self.done_paths.append(target_point_index)
        self.total_path_length = self.total_path_length + displacement #總路徑長
        self.total_steps = self.total_steps + steps #執行總步數(時間)
        #-----------------------------------------
        next_state = self.get_state()
        reward = self.calculate_reward(repeat_selection)
        self.episode_reward = self.episode_reward + reward
        if set(self.taget_start_points).issubset(set(self.done_paths)):
            self.done = True  #當所有的路徑跑完了，done=True
        truncated=False
        info={}
        if self.done or truncated:
            info["episode reward"] = self.episode_reward
            info["total path length"] = self.total_path_length
            info["total_steps"] = self.total_steps
            info["finish_path_num"] = self.finish_path_num

            # print("episode_reward = ",self.episode_reward)
            # print("total_path_length = ",self.total_path_length)
            # print("total_steps = ",self.total_steps)

        return next_state, reward, self.done, truncated, info

    