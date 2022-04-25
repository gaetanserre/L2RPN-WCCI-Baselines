# %%
import grid2op
import numpy as np
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from grid2op.Parameters import Parameters
from grid2op.utils import ScoreL2RPN2020
import torch
import datetime
import sys
import re

from utils import *

from examples.ppo_stable_baselines.B_train_agent import CustomReward

# %%
ENV_NAME = "l2rpn_wcci_2022_dev"

# Split sets and statistics parameters
is_windows = sys.platform.startswith("win32")
is_windows_or_darwin = sys.platform.startswith("win32") or sys.platform.startswith("darwin")
nb_process_stats = 4 if not is_windows_or_darwin else 1
deep_copy = is_windows  # force the deep copy on windows (due to permission issue in symlink in windows)
verbose = 1
SCOREUSED = ScoreL2RPN2020  # ScoreICAPS2021
name_stats = "_reco_powerline"

# Train parameters
env_name_train = '_'.join([ENV_NAME, "train"])
save_path = "./saved_model"
name = '_'.join(["GymEnvWithRecoWithDN", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')])
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

# Chronix to use
def filter_chronics(x):
  list_chronics = ['2050-01-10_0', '2050-08-01_7'] # Names of chronics to keep
  p = re.compile(".*(" + '|'.join([c + '$' for c in list_chronics]) + ")")
  return re.match(p, x) is not None

# %%
# Generate statistics

# env = grid2op.make(ENV_NAME)

try:
  # nm_train, nm_val, nm_test = split_train_val_test_sets(env, deep_copy)
  # generate_statistics([nm_val, nm_test], SCOREUSED, nb_process_stats, name_stats, verbose)
  generate_statistics([ENV_NAME], SCOREUSED, nb_process_stats, name_stats, verbose, filter_fun=filter_chronics)
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
train_args["iterations"] = 700_000
train_args["learning_rate"] = 3e-4
train_args["net_arch"] = [200, 200, 200, 200]
train_args["gamma"] = 0.999
train_args["gymenv_kwargs"] = {"safe_max_rho": 0.9}
train_args["normalize_act"] = True
train_args["normalize_obs"] = True

train_args["save_every_xxx_steps"] = min(train_args["iterations"] // 10, 100_000)

train_args["n_steps"] = 256
train_args["batch_size"] = 64


# %%
p = Parameters()
p.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True # It causes errors during training

# env_train = grid2op.make(env_name_train,
#                    reward_class=CustomReward,
#                    backend=LightSimBackend(),
#                    chronics_class=MultifolderWithCache,
#                    param=p)

env_train = grid2op.make(ENV_NAME,
                   reward_class=CustomReward,
                   backend=LightSimBackend(),
                   chronics_class=MultifolderWithCache,
                   param=p)
env_train.chronics_handler.real_data.set_filter(filter_chronics)
env_train.chronics_handler.real_data.reset()

values_to_test = np.array([3e-5, 3e-4, 3e-3])
var_to_test = "learning_rate"
agents = iter_hyperparameters(env_train, train_args, name, var_to_test, values_to_test)

# %%
env_name_val = '_'.join([ENV_NAME, "val"])
for i, (agent_name, _) in enumerate(agents):
  results = eval_agent(ENV_NAME, #env_name_val,
            2,
            agent_name,
            save_path,
            SCOREUSED,
            gymenv_class,
            verbose,
            gymenv_kwargs=train_args["gymenv_kwargs"] if var_to_test!="gymenv_kwargs" else values_to_test[i],
            param=p,
            filter_fun=filter_chronics)
  print(results)
