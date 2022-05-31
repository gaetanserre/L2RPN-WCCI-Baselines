# %%
import grid2op
import numpy as np
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache
from grid2op.Parameters import Parameters
from grid2op.utils import ScoreL2RPN2020
import torch
import datetime
import sys
import re

from utils import *
from CustomGymEnv import CustomGymEnv

from examples.ppo_stable_baselines.B_train_agent import CustomReward

# %%
ENV_NAME = "l2rpn_wcci_2022"

# Split sets and statistics parameters
is_windows = sys.platform.startswith("win32")
is_windows_or_darwin = is_windows or sys.platform.startswith("darwin")
nb_process_stats = 1 if not is_windows_or_darwin else 1
deep_copy = is_windows  # force the deep copy on windows (due to permission issue in symlink in windows)
verbose = 1
SCOREUSED = ScoreL2RPN2020  # ScoreICAPS2021
name_stats = "_reco_powerline"

# Train parameters
env_name_train = '_'.join([ENV_NAME, "train"])
save_path = "./saved_model/lr/"
name = '_'.join(["CustomGymEnv", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')])
gymenv_class = CustomGymEnv


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
# Generate statistics

def filter_chronics(x):
  list_chronics = ["2050-01-03_31",
                   "2050-02-21_31",
                   "2050-03-07_31",
                   "2050-04-18_31"] # Names of chronics to keep
  p = re.compile(".*(" + '|'.join([c + '$' for c in list_chronics]) + ")")
  return re.match(p, x) is not None

try:
  #nm_train, nm_val, nm_test = split_train_val_test_sets(ENV_NAME, deep_copy)
  generate_statistics([ENV_NAME],
                      SCOREUSED,
                      nb_process_stats,
                      name_stats,
                      verbose,
                      filter_fun=filter_chronics)
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
train_args["act_attr_to_keep"] = ["set_storage", "curtail"]
train_args["iterations"] = 700_000
train_args["learning_rate"] = 1e-4
train_args["net_arch"] = [300, 300, 300]
train_args["gamma"] = 0.999
train_args["gymenv_kwargs"] = {"safe_max_rho": 0.1}
train_args["normalize_act"] = True
train_args["normalize_obs"] = True

train_args["save_every_xxx_steps"] = min(train_args["iterations"] // 10, 100_000)

train_args["n_steps"] = 16
train_args["batch_size"] = 16

# %%
p = Parameters()
p.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True

env_train = grid2op.make(ENV_NAME,
                   reward_class=CustomReward,
                   backend=LightSimBackend(),
                   chronics_class=MultifolderWithCache,
                   param=p)

if filter_chronics is not None:
  env_train.chronics_handler.real_data.set_filter(filter_chronics)
  env_train.chronics_handler.real_data.reset()

values_to_test = np.array([1e-5, 3e-5, 1e-4])
var_to_test = "learning_rate"
agents = iter_hyperparameters(env_train, train_args, name, var_to_test, values_to_test)
