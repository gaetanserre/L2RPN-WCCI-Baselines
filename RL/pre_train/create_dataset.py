
import sys
sys.path.insert(0, "../")

import os
import grid2op
from l2rpn_baselines.OptimCVXPY import OptimCVXPY
from lightsim2grid import LightSimBackend
from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch

torch.cuda.set_device(1)

os.chdir('/home/boguslawskieva/L2RPN-WCCI-Baselines/RL')

env_name = "l2rpn_wcci_2022_dev_train"  # name subject to change
is_test = False
datetime_now=datetime.now().strftime('%Y-%m-%d_%H-%M')
NUM_CHRONICS = 100

env = grid2op.make(env_name,
                   test=is_test,
                   backend=LightSimBackend()
                   )
env.chronics_handler.real_data.shuffle()

rho_safe   = 0.6
rho_danger = 0.97
agent = OptimCVXPY(env.action_space,
                   env,
                   rho_danger=rho_danger,
                   rho_safe=rho_safe,
                   penalty_redispatching_unsafe=10,
                   penalty_storage_unsafe=0.04,
                   penalty_curtailment_unsafe=0.04,
                   margin_th_limit=0.93,
                   alpha_por_error=0.5
                   )


expert_observations = []
expert_actions = []
expert_flag = [] # 0 for safe, 2 for unsafe, 1 otherwise
for num in range(NUM_CHRONICS):
    obs = env.reset()
    print("Current chronic:", env.chronics_handler.get_name())
    agent.reset(obs)
    done = False
    for nb_step in tqdm(range(obs.max_step)):
        prev_obs = obs
        act = agent.act(obs)
        expert_observations.append(prev_obs.to_vect())
        expert_actions.append(act)
        if obs.rho.max() > rho_danger:
            expert_flag.append(2)
        elif obs.rho.max() < rho_safe:
            expert_flag.append(0)
        else:
            expert_flag.append(1)
        obs, reward, done, info = env.step(act)
        if done:
            break
    print(f"\t scenario: {os.path.split(env.chronics_handler.get_id())[-1]}: {nb_step + 1} / {obs.max_step}")

    np.savez_compressed(
        "pre_train/expert_data/expert_data_{}".format(datetime_now),
        expert_actions=expert_actions,
        expert_observations=expert_observations,
        expert_flag=expert_flag,
    )