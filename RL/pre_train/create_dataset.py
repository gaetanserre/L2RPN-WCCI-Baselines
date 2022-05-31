# https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pretraining.ipynb

import sys
sys.path.insert(0, "../")

import os
import grid2op
from l2rpn_baselines.OptimCVXPY import OptimCVXPY
from lightsim2grid import LightSimBackend
from tqdm import tqdm
import numpy as np

env_name = "l2rpn_wcci_2022_dev_train"  # name subject to change
is_test = False
NUM_CHRONICS = 100

env = grid2op.make(env_name,
                   test=is_test,
                   backend=LightSimBackend()
                   )

rho_safe   = 0.85
rho_danger = 0.95
agent = OptimCVXPY(env.action_space,
                   env,
                   rho_danger=rho_danger,
                   rho_safe=rho_safe,
                   penalty_redispatching_unsafe=0.99,
                   penalty_storage_unsafe=0.01,
                   penalty_curtailment_unsafe=0.01,
                   )


expert_observations = []
expert_actions = []
for num in range(NUM_CHRONICS):
  obs = env.reset()
  print("Current chronic:", env.chronics_handler.get_name())
  agent.reset(obs)
  done = False
  for nb_step in tqdm(range(obs.max_step)):
    prev_obs = obs
    act = agent.act(obs)
    if obs.rho.max() >= rho_danger:
      expert_observations.append(prev_obs.to_vect())
      expert_actions.append(act)
    obs, reward, done, info = env.step(act)
    if done:
      break
  print(f"\t scenario: {os.path.split(env.chronics_handler.get_id())[-1]}: {nb_step + 1} / {obs.max_step}")

  np.savez_compressed(
    "expert_data",
    expert_actions=expert_actions,
    expert_observations=expert_observations,
  )