from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn

archs = [[1], [10], [64, 64], [128, 128], [256, 64]]
#envs = ["Pendulum-v1", "CarRacing-v0", "MountainCarContinuous-v0"]
#names = ["pendulum", "car_racing", "moutaincar_c0"]
envs = ["BipedalWalker-v3"]
names = ["bipedalwalker"]
ent_coeffs = [0.0, 0.3]

for arch in archs:
  dir_name = "ppo-nn-" + "-".join([str(i) for i in arch])
  policy_kwargs = dict(activation_fn=nn.Tanh,
                      net_arch=[dict(pi=arch, vf=arch)])
  for i in range(len(envs)):
    for ent_coef in ent_coeffs:
      if ent_coef != 0.0:
        name = "exps/" + dir_name + "/ppo_entropy_" + names[i]
        tensorboard_log = "logs/" + names[i] + "_entropy"
      else:
        name = "exps/" + dir_name + "/ppo_" + names[i]
        tensorboard_log = "logs/" + names[i]

      env = make_vec_env(envs[i], n_envs=8)
      model = PPO(policy="MlpPolicy",
                  env=env,
                  ent_coef=ent_coef,
                  verbose=1,
                  policy_kwargs=policy_kwargs,
                  tensorboard_log=tensorboard_log)
      model.learn(total_timesteps=1_000_000)
      model.save(name)
