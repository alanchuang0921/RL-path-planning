# from ddpg_pendulum import *
from webots_grinding_env import World
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

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("CUDA is not available, falling back to CPU.")


class CustomTensorboardCallback(BaseCallback):
    def __init__(self, log_freq=500,ep_reward_log_freq=5, verbose=0):
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
                self.writer.add_scalar("episode/episode_reward", info["episode reward"], timestep)

            if "success rate" in info:
                self.writer.add_scalar("success rate", info["success rate"], timestep)     

            if "get_to_target_times" in info:
                self.writer.add_scalar("get_to_target_times", info["get_to_target_times"], timestep)

            if "min errors" in info:
                for i, e in enumerate(info["min errors"]):
                    self.writer.add_scalar(f"min errors/error{i+1}", e, timestep)

            for key, value in info.items():
                if key.startswith("min pos errors/face_"):
                    self.writer.add_scalar(key, value, timestep)

            for key, value in info.items():
                if key.startswith("min orien errors/face_"):
                    self.writer.add_scalar(key, value, timestep)
        # 只有每 log_freq 個 timestep 才寫入
        if timestep % self.log_freq == 0:    
            if "timestep reward" in info:
                self.writer.add_scalar("timestep reward", info["timestep reward"], timestep)

            if "timestep rewards" in info:
                for i, r in enumerate(info["timestep rewards"]):
                    self.writer.add_scalar(f"rewards/r{i+1}", r, timestep)

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

world = World()
timestep = int(world.timestep)# 設定時鐘（time step）
policy_kwargs = dict(log_std_init=1.0)

model = PPO(
    policy = "MlpPolicy", 
    env= world, 
    device="cuda",
    batch_size=512,
    learning_rate= 3e-4,  #預設3e-4
    n_steps= 2048
    # policy_kwargs=policy_kwargs
    # 增加探索性
    # ent_coef=0.02              # ✅ 增加熵來鼓勵策略的隨機性
    # gamma=0.95                 # ✅ 減少折扣因子，重視短期獎勵
    # gae_lambda=0.9              # ✅ 減少 GAE 平滑，增加估值變動性
    )

# 設定訓練參數與 checkpoint callback
# steps = 2048
# episodes = 100000
# total_timesteps = steps * episodes
total_timesteps = 4000000
file_name="_4faceS_wp7cm_changWithSuccessRate_noCrashDone_a1_20"

checkpoint_callback = CheckpointCallback(
    save_freq=total_timesteps  // 20,  # 每訓練 1% 儲存一次
    save_path="./models/",
    name_prefix=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ file_name
)
callback = CallbackList([
    checkpoint_callback,
    CustomTensorboardCallback()  # 你自己寫的 callback
])

# model_path = "models/ppo2_train_4096000_steps"  # <-- 替換成你自己的模型路徑
# model_path = "models/20250613-155624_4faceS_wp7cm_crash_keeptrain_12288000_steps"
# model = PPO.load(model_path, env=world, device="cuda")  # 如果有用 GPU 的話
# print("Policy on device:", model.policy.device)
model.learn(total_timesteps= total_timesteps,  callback=callback)   

########################################################### 結果測試 #############################################################

# # 載入之前訓練好的模型

# env = DummyVecEnv([lambda: world])          # SB3 需要 VecEnv 格式

# # 載入已訓練的 PPO 模型
# model_path = "models/20250530-203151_allface_32768000_steps"  # <-- 替換成你自己的模型路徑
# model = PPO.load(model_path, env=env, device="cuda")  # 如果有用 GPU 的話

# # 重設環境
# obs = env.reset()

# # 開始執行測試 Episode
# done = False
# episode_reward = 0
# step_count = 0

# i=0
# start_points = world.get_init_point_on_path()
# print("start_points = ",start_points)


# while not done:
#     samplepoint_num = start_points[i]
#     world.select_samplepoint_num(samplepoint_num)
#     action, _ = model.predict(obs, deterministic=True)  # ✅ 採用最優策略
#     obs, reward, done, info = env.step(action)
#     episode_reward += reward[0]  # reward 是 list（VecEnv 包裝）
#     step_count += 1

#     get_to_target = info[0]["get to target"]

#     if get_to_target:# 如果達到目標位置，更改下一個目標位置
#         i = i+1
#     if done:
#         print("obs=",obs)
#     print(done)


# print(f"✅ 測試完成，共 {step_count} 步，總回報為：{episode_reward:.2f}")





