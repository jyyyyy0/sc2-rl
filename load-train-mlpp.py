# $ source ~/Desktop/sc2env/bin/activate

# so this works, so far. 

import torch

from stable_baselines3 import PPO
import os
from sc2env import Sc2Env
import time
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.reward_sum = 0.0
        self.num_steps = 0

    def _on_step(self) -> bool:
        # 从环境中获取奖励
        reward = self.training_env.get_attr('reward')[0]
        self.reward_sum += reward
        self.num_steps += 1

        return True

    def _on_rollout_end(self) -> None:
        # 计算平均奖励并打印
        average_reward = self.reward_sum / self.num_steps
        print("平均奖励: ", average_reward)
        # 重置计数器
        self.reward_sum = 0.0
        self.num_steps = 0

# 使用自定义回调
custom_callback = CustomCallback(verbose=1)
LOAD_MODEL = "models/1713099092/10000.zip"
# Environment:
env = Sc2Env()

# load the model:
model = PPO.load(LOAD_MODEL, env=env)

model_name = f"{int(time.time())}"

models_dir = f"models/{model_name}/"
logdir = f"logs/{model_name}/"


conf_dict = {"Model": "load-v16s",
             "Machine": "Puget/Desktop/v18/2",
             "policy":"MlpPolicy",
             "model_save_name": model_name,
             "load_model": LOAD_MODEL
             }

run = wandb.init(
    project=f'SC2RLv8',
    entity="jyyyy-1",
    config=conf_dict,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    save_code=True,  # save source code
)


# further train:
TIMESTEPS = 20000
iters = 0
max_iterations = 5  # 设置最大迭代次数
should_stop = False

while not should_stop and iters < max_iterations:

    print("On iteration: ", iters)
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO", callback=[WandbCallback(verbose=2), custom_callback])
    model.save(f"{models_dir}/{TIMESTEPS * iters}")

    # 检查结束条件
    if iters >= max_iterations:
        should_stop = True
