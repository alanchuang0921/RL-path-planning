# from ddpg_pendulum import *
from webots_grinding_env import World
from path_connect import path_select_env
from path_connect import path_connect_env
# from ddpg import DDPG as Agent
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

# # 載入之前訓練好的模型

env = DummyVecEnv([lambda: world])          # SB3 需要 VecEnv 格式

# # 載入已訓練的 PPO 模型
model_path = "models/20250623-230114_select_startpoint_byPathLength_noDone_100000_steps"  # <-- 替換成你自己的模型路徑
model = PPO.load(model_path, env=env, device="cuda")  # 如果有用 GPU 的話

# # 重設環境
obs = env.reset()

# 開始執行測試 Episode
done = False
episode_reward = 0
step_count = 0
path_order = []
while not done:
    action, _ = model.predict(obs, deterministic=True)  # ✅ 採用最優策略
    path_order.append(action[0])
    obs, reward, done, info = env.step(action)
    episode_reward += reward[0]  # reward 是 list（VecEnv 包裝）
    step_count += 1

print(path_order)
# print(f"✅ 測試完成，共 {step_count} 步，總回報為：{episode_reward:.2f}")
#---------------------------------------
# env=path_select_env()

# world = path_connect_env()
# env.reset()
# samplepoint_num=21
# world.execute_task(samplepoint_num)
#----------------------------------------
