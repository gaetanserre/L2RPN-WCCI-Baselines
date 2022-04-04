# %%
import grid2op
import numpy as np
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from grid2op.Parameters import Parameters
from grid2op.utils import ScoreL2RPN2020
import torch

from utils import *

import sys
sys.path.insert(0, "examples/")

from ppo_stable_baselines.B_train_agent import CustomReward

# %%
ENV_NAME = "l2rpn_wcci_2022_dev"

# Split sets and statistics parameters
is_windows = sys.platform.startswith("win32")
nb_process_stats = 1 if not is_windows else 1
deep_copy = is_windows  # force the deep copy on windows (due to permission issue in symlink in windows)
verbose = 1
SCOREUSED = ScoreL2RPN2020  # ScoreICAPS2021
name_stats = "_reco_powerline"

# Train parameters
train_env_name = "l2rpn_wcci_2022_dev_train"
save_path = "./saved_model"
name = "expe_GymEnvWithRecoWithDN_2022"
gymenv_class = GymEnvWithRecoWithDN


# %%
train_args = {}

# Utility parameters PPO
train_args["logs_dir"] = "./logs"
train_args["save_path"] = save_path
train_args["name"] = name
train_args["verbose"] = 1
train_args["gymenv_class"] = gymenv_class
train_args["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
env = grid2op.make(ENV_NAME)

try:
  nm_train, nm_val, nm_test = split_train_val_test_sets(env, deep_copy)
  generate_statistics(nm_val, nm_test, SCOREUSED, nb_process_stats, name_stats, verbose)
except Exception as e:
  if str(e).startswith("Impossible to create"):
    pass
  else:
    raise e

# %%
# Learn parameters PPO
train_args["obs_attr_to_keep"] = ["month", "day_of_week", "hour_of_day", "minute_of_hour",
                                  "gen_p", "load_p", 
                                  "p_or", "rho", "timestep_overflow", "line_status",
                                  # dispatch part of the observation
                                  "actual_dispatch", "target_dispatch",
                                  # storage part of the observation
                                  "storage_charge", "storage_power",
                                  # curtailment part of the observation
                                  "curtailment", "curtailment_limit",  "gen_p_before_curtail",
                                  ]
train_args["act_attr_to_keep"] = ["curtail", "set_storage"]
train_args["iterations"] = 20
train_args["learning_rate"] = 3e-4
train_args["net_arch"] = [200, 200, 200, 200]
train_args["gamma"] = 0.999
train_args["gymenv_kwargs"] = {"safe_max_rho": 0.9}
train_args["normalize_act"] = True
train_args["normalize_obs"] = True

train_args["save_every_xxx_steps"] = min(train_args["iterations"] // 10, 100_000)

train_args["n_steps"] = 10
train_args["batch_size"] = 5


# %%
p = Parameters()
# p.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True It causes errors during training

env = grid2op.make(train_env_name,
                   reward_class=CustomReward,
                   backend=LightSimBackend(),
                   chronics_class=MultifolderWithCache,
                   param=p)

lr_values = np.array([3e-4, 0.2])
agents = iter_hyperparameters(env, train_args, name, "learning_rate", lr_values)

# %%
env_name = "l2rpn_wcci_2022_dev_val"
results = eval_agent(env_name,
           2,
           agents[0][0],
           save_path,
           SCOREUSED,
           nb_process_stats,
           gymenv_class,
           verbose)
print(results)
