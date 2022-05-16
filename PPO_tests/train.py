from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn

# Parallel environments
""" env = make_vec_env("CartPole-v1", n_envs=8)
model = PPO(policy="MlpPolicy", env=env, verbose=1, tensorboard_log="logs/cartpole")
model.learn(total_timesteps=1_000_000)
model.save("ppo_cartpole")

env = make_vec_env("Acrobot-v1", n_envs=8)
model = PPO(policy="MlpPolicy", env=env, verbose=1, tensorboard_log="logs/acrobot")
model.learn(total_timesteps=1_000_000)
model.save("ppo_acrobot")

env = make_vec_env("MountainCar-v0", n_envs=8)
model = PPO(policy="MlpPolicy", env=env, verbose=1, tensorboard_log="logs/mountaincar")
model.learn(total_timesteps=1_000_000)
model.save("ppo_mountaincar") """

archs = [[1], [10], [64, 64]]
for arch in archs:
  dir_name = "ppo-nn-" + "-".join([str(i) for i in arch])
  policy_kwargs = dict(activation_fn=nn.ReLU,
                      net_arch=[dict(pi=arch, vf=arch)])

  env = make_vec_env("Pendulum-v1", n_envs=8)
  model = PPO(policy="MlpPolicy",
              env=env,
              verbose=1,
              policy_kwargs=policy_kwargs,
              tensorboard_log="logs/pendulum")
  model.learn(total_timesteps=1_000_000)
  model.save("exps/" + dir_name + "/ppo_pendulum")

  env = make_vec_env("MountainCarContinuous-v0", n_envs=8)
  model = PPO(policy="MlpPolicy",
              env=env,
              verbose=1,
              policy_kwargs=policy_kwargs,
              tensorboard_log="logs/mountaincar_c0")
  model.learn(total_timesteps=1_000_000)
  model.save("exps/" + dir_name + "/ppo_mountaincar_c0")
