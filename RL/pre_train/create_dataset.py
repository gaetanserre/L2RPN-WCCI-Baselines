
import sys
sys.path.insert(0, "../")

import os
import json
import grid2op
from l2rpn_baselines.OptimCVXPY import OptimCVXPY
from lightsim2grid import LightSimBackend
from tqdm import tqdm
import numpy as np
from datetime import datetime
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, GymEnv
from l2rpn_baselines.PPO_SB3.utils import remove_non_usable_attr
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
import re

# Create grid2op env
env_name = "l2rpn_wcci_2022_train"  # name subject to change
is_test = False
datetime_now=datetime.now().strftime('%Y-%m-%d_%H-%M')
NUM_CHRONICS = 3

env = grid2op.make(env_name,
                   test=is_test,
                   backend=LightSimBackend())

def filter_chronics(x):
  list_chronics = ["2050-01-03_1"]
  p = re.compile(".*(" + '|'.join([c + '$' for c in list_chronics]) + ")")
  return re.match(p, x) is not None

env.chronics_handler.real_data.set_filter(filter_chronics)
env.chronics_handler.real_data.reset()
env.chronics_handler.real_data.shuffle()

# Create gym env
with open("./preprocess_obs.json", "r", encoding="utf-8") as f:
    obs_space_kwargs = json.load(f)
with open("./preprocess_act.json", "r", encoding="utf-8") as f:
    act_space_kwargs = json.load(f)

obs_attr_to_keep = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                                  "gen_p", "load_p", 
                                  "p_or", "rho", "timestep_overflow", "line_status",
                                  # dispatch part of the observation
                                  "actual_dispatch", "target_dispatch",
                                  # storage part of the observation
                                  "storage_charge", "storage_power",
                                  # curtailment part of the observation
                                  "curtailment", "curtailment_limit",  "gen_p_before_curtail",
                                  ]
act_attr_to_keep = ["curtail", "set_storage"]
act_attr_to_keep = remove_non_usable_attr(env, act_attr_to_keep)


gymenv_kwargs = {}
gymenv_class = GymEnvWithRecoWithDN
env_gym = gymenv_class(env, **gymenv_kwargs)
env_gym.observation_space.close()
env_gym.observation_space = BoxGymObsSpace(env.observation_space,
                                        attr_to_keep=obs_attr_to_keep,
                                        **obs_space_kwargs)
env_gym.action_space.close()
env_gym.action_space = BoxGymActSpace(env.action_space,
                                    attr_to_keep=act_attr_to_keep,
                                    **act_space_kwargs)

for attr_nm in act_attr_to_keep:
            if (("multiply" in act_space_kwargs and attr_nm in act_space_kwargs["multiply"]) or 
                ("add" in act_space_kwargs and attr_nm in act_space_kwargs["add"]) 
               ):
                # attribute is scaled elsewhere
                continue
            env_gym.action_space.normalize_attr(attr_nm)

for attr_nm in obs_attr_to_keep:
            if (("divide" in obs_space_kwargs and attr_nm in obs_space_kwargs["divide"]) or 
                ("subtract" in obs_space_kwargs and attr_nm in obs_space_kwargs["subtract"]) 
               ):
                # attribute is scaled elsewhere
                continue
            env_gym.observation_space.normalize_attr(attr_nm)

def to_gym_act(env_gym, g2op_actions):
    converted_acts = []
    for g2op_act in g2op_actions:
        gym_act = np.zeros((0,))
        for attr_nm in env_gym.action_space._attr_to_keep :
            this_part_act = getattr(g2op_act, attr_nm).copy()
            if attr_nm == "curtail" or attr_nm == "curtail_mw":
                this_part_act = this_part_act[env_gym.init_env.action_space.gen_renewable]
            if attr_nm == "redispatch":
                this_part_act = this_part_act[env_gym.init_env.action_space.gen_redispatchable]
            this_part_act = (this_part_act - env_gym.action_space._add[attr_nm]) / env_gym.action_space._multiply[attr_nm]
            gym_act = np.concatenate((gym_act, this_part_act))
        converted_acts.append(gym_act)
    return converted_acts

# Define optimizer agent
rho_safe   = 0.95
rho_danger = 0.97

""" agent = OptimCVXPY(env.action_space,
                   env,
                   penalty_redispatching_unsafe=0.99,
                   penalty_storage_unsafe=0.01,
                   penalty_curtailment_unsafe=0.01,
                   rho_safe=rho_safe,
                   rho_danger=rho_danger,
                   margin_th_limit=0.93,
                   alpha_por_error=0.5,
                   weight_redisp_target=0.3) """

agent = OptimCVXPY(env.action_space,
                   env,
                   penalty_redispatching_unsafe=0.01,
                   penalty_storage_unsafe=0.01,
                   penalty_curtailment_unsafe=0.01,
                   rho_safe=rho_safe,
                   rho_danger=rho_danger,
                   margin_th_limit=0.93,
                   alpha_por_error=0.5,
                   weight_redisp_target=0.3)


expert_observations = []
expert_actions = []
expert_gym_actions = []
expert_flag = [] # 0 for safe, 2 for unsafe, 1 otherwise
for num in range(NUM_CHRONICS):
    obs = env.reset()
    print("Current chronic:", env.chronics_handler.get_name())
    agent.reset(obs)
    done = False
    for nb_step in tqdm(range(obs.max_step)):
        prev_obs = obs
        act = agent.act(obs)
        expert_observations.append(prev_obs.to_vect()) # normaliser observation d'abord
        expert_actions.append(act) # tronquer et normalier action d'abord
        expert_gym_actions.append(to_gym_act(env_gym, [act])[0])
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
        "expert_data/expert_data_{}".format(datetime_now),
        expert_actions=expert_actions,
        expert_observations=expert_observations,
        expert_flag=expert_flag,
        expert_gym_actions=expert_gym_actions
    )