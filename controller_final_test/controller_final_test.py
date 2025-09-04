# from ddpg_pendulum import *
from webots_grinding_env import World
from path_connect import path_select_env
from path_connect import path_connect_env
# from ddpg import DDPG as Agent
from controller import Robot, Supervisor,Connector
from ikpy.chain import Chain
from gym.spaces import Box
import tensorflow as tf
import random
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
import pandas as pd 
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import DQN
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("CUDA is not available, falling back to CPU.")


class CustomTensorboardCallback(BaseCallback):
    def __init__(self, log_freq=500,ep_reward_log_freq=2, verbose=0):
        super(CustomTensorboardCallback, self).__init__(verbose)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ file_name
        # self.writer = SummaryWriter(log_dir="./ppo_tensorboard_log")
        self.writer = SummaryWriter(log_dir)
        self.log_freq = log_freq
        self.ep_reward_log_freq = ep_reward_log_freq

    def _on_step(self) -> bool:
        
        timestep = self.num_timesteps
        info = self.locals.get("infos", [{}])[0]  # 取第一個環境的 info

        # 更頻繁記錄 episode reward
        if timestep % self.ep_reward_log_freq == 0 :
            if "episode reward" in info:
                self.writer.add_scalar("episode reward", info["episode reward"], timestep)
            if "total path length" in info:
                self.writer.add_scalar("total path length", info["total path length"], timestep)
            if "total_steps" in info:
                self.writer.add_scalar("total_steps", info["total_steps"], timestep)   
            if "finish_path_num" in info:
                self.writer.add_scalar("finish_path_num", info["finish_path_num"], timestep)                 

            # if "memory used" in info:
            #     self.writer.add_scalar("memory used", info["memory used"], timestep)       

        return True

    # def _on_rollout_end(self) -> None:
    #     # 這裡可以計算每一個 episode 的平均 reward
    #     ep_rewards = self.locals.get("ep_info_buffer", [])
    #     if len(ep_rewards) > 0:
    #         mean_ep_reward = sum([ep["r"] for ep in ep_rewards]) / len(ep_rewards)
    #         self.writer.add_scalar("episode/mean_reward", mean_ep_reward, self.num_timesteps)

    def _on_training_end(self) -> None:
        self.writer.close()




#若要檢視訓練結果，進入webots環境後，在terminal輸入tensorboard --logdir=儲存數據資料夾的絕對路徑
#輸入：tensorboard --logdir=./ppo_tensorboard_log/
################################################################################################################################
"""
訓練如何決策路徑串接順序
"""
world = path_select_env()
# timestep = int(world.timestep)# 設定時鐘（time step）
policy_kwargs = dict(log_std_init=1.0)

# model = DQN(
#     policy="MlpPolicy",
#     env=world,
#     learning_rate=1e-3,
#     buffer_size=50000,
#     learning_starts=1000,
#     batch_size=32,
#     gamma=0.99,
#     train_freq=4,
#     target_update_interval=1000,
#     verbose=1,
# )
model = PPO(
    policy="MlpPolicy",
    env=world,
    learning_rate=3e-4,
    batch_size=64,
    n_steps=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
)
# 設定訓練參數與 checkpoint callback
steps = 1000
episodes = 100
total_timesteps = steps * episodes
file_name="_select_startpoint_byPathLength_noDone"


checkpoint_callback = CheckpointCallback(
    save_freq=steps * episodes // 5,  # 每訓練 20% 儲存一次
    save_path="./models/",
    name_prefix=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ file_name
)
callback = CallbackList([
    checkpoint_callback,
    CustomTensorboardCallback()  # 你自己寫的 callback
])

# model_path = "models/20250623-230114_select_startpoint_byPathLength_noDone_100000_steps"  # <-- 替換成你自己的模型路徑
# model = PPO.load(model_path, env=world, device="cuda")  # 如果有用 GPU 的話
# print("Policy on device:", model.policy.device)
# model.learn(total_timesteps= steps*episodes,  callback=callback)   

################################################################################################################################
"""
測試路徑串接順序
"""
# # 載入之前訓練好的模型

# env = DummyVecEnv([lambda: world])          # SB3 需要 VecEnv 格式

# # # 載入已訓練的 PPO 模型
# model_path = "models/20250623-230114_select_startpoint_byPathLength_noDone_100000_steps"  # <-- 替換成你自己的模型路徑
# model = PPO.load(model_path, env=env, device="cuda")  # 如果有用 GPU 的話

# # # 重設環境
# obs = env.reset()

# # 開始執行測試 Episode
# done = False
# episode_reward = 0
# step_count = 0
# path_order = []
# while not done:
#     action, _ = model.predict(obs, deterministic=True)  # ✅ 採用最優策略
#     path_order.append(action[0])
#     obs, reward, done, info = env.step(action)
#     episode_reward += reward[0]  # reward 是 list（VecEnv 包裝）
#     step_count += 1

# print(path_order)
# print(f"✅ 測試完成，共 {step_count} 步，總回報為：{episode_reward:.2f}")

################################################################################################################################
"""
測試路徑串接：
隨機選擇500條路徑順序
紀錄總路徑長以及誤差
"""
world = path_select_env()
start_points = world.taget_start_points
n = 0
test_times = 1
results = []
success_count = 0  # 成功次數統計
numbers = list(range(0, len(start_points))) #採樣點的順序
while n < test_times:
    random.shuffle(numbers) 
    numbers = [2, 3, 4, 5, 6, 7, 0, 1, 10, 9, 8] # 經過路徑順序優化後所產生的順序

    # print("numbers =",numbers)
    done = False
    i=0
    world.reset()
    while not done:
        number=numbers[i]
        next_state, reward, done, truncated, info = world.step(number)
        i=i+1
    # print("info[crash]=",info["crash"])

    if info["crash"]==False and done==True:#沒有發生碰撞
        success_count += 1  # 記錄成功次數

        position_errors,orientation_errors = world.get_pos_ori_errors()
        if len(position_errors) > 0 and len(orientation_errors):
            average_position_errors = sum(position_errors) / len(position_errors)
            pos_std_dev = np.std(position_errors)
            average_orientation_errors = sum(orientation_errors) / len(orientation_errors)
            ori_std_dev = np.std(orientation_errors)
            total_path_length = world.total_path_length
            finish_path_num = world.finish_path_num

            # print("position_errors_avg = ",average_position_errors)
            # print("position_errors_std = ",pos_std_dev)
            # print("max_position_errors = ",max(position_errors))

            # print("orientation_errors_avg = ",average_orientation_errors )
            # print("orientation_errors_astd = ",ori_std_dev)
            # print("max_orientation_errors = ",max(orientation_errors))

            # print("total_path_length = ",total_path_length )

            # 加入結果列表
            results.append({
                "test_num": n,
                "finish_path_num":finish_path_num,
                "average_position_error": average_position_errors,
                "position_error_std": pos_std_dev,
                "max_position_error": max(position_errors),
                "average_orientation_error": average_orientation_errors,
                "orientation_error_std": ori_std_dev,
                "max_orientation_error": max(orientation_errors),
                "total_path_length": total_path_length

            })

    n += 1 

print(f"Total tests: {test_times}")
print(f"Success count: {success_count}")
success_rate = success_count / test_times
print(f"Success rate: {success_rate:.2%}")

print("total_path_length = ",total_path_length)
# 轉換成DataFrame並存成CSV
"""
紀錄平移、旋轉誤差、總路徑長
"""
# df = pd.DataFrame(results)
# df.to_csv("results_data/path_evaluation_results_0703.csv", index=False)
# print("已儲存結果至 path_evaluation_results.csv")

"""
紀錄路徑上機械手臂的連續姿態
"""
# csv_file_path = "robot_path_csv/joint_poses_4faceS_wp7cm_0703.csv"
# world.save_robot_posture_as_csv(csv_file_path) # 儲存機器手臂的路徑

"""
紀錄機器手臂個軸馬達轉速與時間關係
"""
joint_velocity_file_path = "robot_velocity_csv/joint_velocity.csv"
world.save_joint_velocity_as_csv(joint_velocity_file_path)