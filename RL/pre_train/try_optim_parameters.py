import sys
from tkinter import EXCEPTION
sys.path.insert(0, "../")

import os
import grid2op
from l2rpn_baselines.OptimCVXPY import OptimCVXPY
from lightsim2grid import LightSimBackend
from grid2op.utils import ScoreL2RPN2020
import numpy as np
from utils import *
from tqdm import tqdm
import pdb
import re
import copy
from datetime import datetime

save_path = "./saved_model/"
ENV_NAME = "l2rpn_wcci_2022"
SCOREUSED = ScoreL2RPN2020
verbose = False
p = Parameters()
p.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
datetime_now=datetime.now().strftime('%Y-%m-%d_%H-%M')
is_windows_or_darwin = sys.platform.startswith("win32") or sys.platform.startswith("darwin")
nb_process_stats = 4 if not is_windows_or_darwin else 1
  
train_args = {}
train_args["gymenv_kwargs"] = {"safe_max_rho": 0.2}

def filter_chronics(x):
    list_chronics = ["2050-01-03_31",
                    "2050-02-21_31",
                    "2050-03-07_31",
                    "2050-04-18_31",
                    "2050-05-09_31",
                    "2050-06-27_31",
                    "2050-07-25_31",
                    "2050-08-01_31",
                    "2050-09-26_31",
                    "2050-10-03_31",
                    "2050-11-14_31",
                    "2050-12-19_31",
                    ] # Names of chronics to keep
    p = re.compile(".*(" + '|'.join([c + '$' for c in list_chronics]) + ")")
    return re.match(p, x) is not None

env = grid2op.make(ENV_NAME,
                   backend=LightSimBackend()
                   )
env.chronics_handler.real_data.set_filter(filter_chronics)
env.chronics_handler.real_data.reset()

nb_scenario = 12

default_parameters = {"penalty_redispatching_unsafe":1, # 1, 10 (à lancer cet aprem)
                      "penalty_storage_unsafe":0.01,  # 0.04
                      "penalty_curtailment_unsafe":0.01,
                      "rho_safe":0.6,
                      "rho_danger":0.97, # To tune 0.9, 0.97, 0.3
                      "margin_th_limit":0.93,
                      "alpha_por_error":0.5}


parameters_values = {"penalty_storage_unsafe": [0.01, 0.04],
                     "rho_danger": [0.9, 0.97, 0.3],}
agents_dict = {}
parameters = []

i = 0
for value in parameters_values["penalty_storage_unsafe"]:
  default_parameters["penalty_storage_unsafe"] = value
  for value in parameters_values["rho_danger"]:
    default_parameters["rho_danger"] = value

    agents_dict.update({f"optim_{i}": OptimCVXPY(env.action_space,
                                                env,
                                                **default_parameters
                                                )}
                        )    
    i += 1
    print(default_parameters)
    parameters.append(default_parameters)

print("Start evaluation of agents")

total_results = np.zeros((len(agents_dict), nb_scenario, 3))
for i, (agent_name, my_agent) in enumerate(agents_dict.items()):
    print("Evaluation of : ", agent_name)
    try:
        results = eval_agent(ENV_NAME,
                nb_scenario,
                agent_name,
                save_path,
                SCOREUSED,
                verbose,
                gymenv_kwargs=train_args["gymenv_kwargs"],
                param=p,
                filter_fun=filter_chronics,
                my_agent=my_agent,
                nb_process_stats = nb_process_stats
                )
        for k in range(3):
            total_results[i, :, k]=np.array(results[k])

        parameters[i].update({"datetime_now":datetime_now, "agent_name":agent_name, "agent_id":i})
        with open("./total_results/dicts_optimizers_params.json", 'a') as fp:
            json.dump(parameters[i], fp, indent=4)
    except Exception as e:
        print(e)
        

with open('./total_results/total_results_{}.npy'.format(datetime_now), 'wb') as f:
    np.save(f, total_results)