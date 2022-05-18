from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn

archs = [[1], [10], [64, 64]]
envs = ["Pendulum-v1", "CarRacing-v1"]
names = ["pendulum", "car_racing"]

for arch in archs:
  dir_name = "ppo-nn-" + "-".join([str(i) for i in arch])
  policy_kwargs = dict(activation_fn=nn.ReLU,
                      net_arch=[dict(pi=arch, vf=arch)])
  for i in range(len(envs)):
    env = make_vec_env(envs[i], n_envs=8)
    model = PPO(policy="MlpPolicy",
                env=env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log="logs/" + names[i])
    model.learn(total_timesteps=1_000_000)
    model.save("exps/" + dir_name + "/ppo_" + names[i])
