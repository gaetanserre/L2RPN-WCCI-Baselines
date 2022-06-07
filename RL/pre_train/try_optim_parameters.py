import sys
from tkinter import EXCEPTION
# sys.path.insert(0, "../")
sys.path.insert(0, '/home/boguslawskieva/L2RPN-WCCI-Baselines/RL')

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
import torch

torch.cuda.set_device(2)

os.chdir('/home/boguslawskieva/L2RPN-WCCI-Baselines/RL')

save_path = "./saved_model/"
ENV_NAME = "l2rpn_wcci_2022_dev"
SCOREUSED = ScoreL2RPN2020
verbose = False
p = Parameters()
p.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
datetime_now=datetime.now().strftime('%Y-%m-%d_%H-%M')
is_windows_or_darwin = sys.platform.startswith("win32") or sys.platform.startswith("darwin")
nb_process_stats = 8 if not is_windows_or_darwin else 1
  
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
                    "2050-01-10_31",
                    "2050-02-07_31",
                    "2050-03-14_31",
                    "2050-04-11_31",
                    "2050-05-02_31",
                    "2050-06-20_31",
                    "2050-07-18_31",
                    "2050-08-08_31",
                    "2050-09-19_31",
                    "2050-10-10_31",
                    "2050-11-07_31",
                    "2050-12-12_31",
                    "2050-01-17_31",
                    "2050-02-14_31",
                    "2050-03-21_31",
                    "2050-04-25_31",
                    "2050-05-16_31",
                    "2050-06-13_31",
                    "2050-07-11_31",
                    "2050-08-15_31",
                    "2050-09-12_31",
                    "2050-10-17_31",
                    "2050-11-21_31",
                    "2050-12-05_31",
                    ] # Names of chronics to keep
    p = re.compile(".*(" + '|'.join([c + '$' for c in list_chronics]) + ")")
    return re.match(p, x) is not None

env = grid2op.make(ENV_NAME,
                   backend=LightSimBackend()
                   )
env.chronics_handler.real_data.set_filter(filter_chronics)
env.chronics_handler.real_data.reset()

nb_scenario = 36

# parameters_to_test = [{"penalty_redispatching_unsafe":0, 
#                         "penalty_storage_unsafe":0.01, 
#                         "penalty_curtailment_unsafe":0.01,
#                         "rho_safe":0.95,
#                         "rho_danger":0.97,
#                         "margin_th_limit":0.93,
#                         "alpha_por_error":0.5}#,
#                         {"penalty_redispatching_unsafe":1, # 1, 10 (à lancer cet aprem)
#                         "penalty_storage_unsafe":0.01,  # 0.04
#                         "penalty_curtailment_unsafe":0.01,
#                         "rho_safe":0, # gaetan 0.6
#                         "rho_danger":0.97, # To tune 0.9, 0.97, 0.3
#                         "margin_th_limit":0.93,
#                         "alpha_por_error":0.5}
#                     ]

# parameters_to_test = [{"penalty_redispatching_unsafe":10, 
#                         "penalty_storage_unsafe": penalty_storage_unsafe, 
#                         "penalty_curtailment_unsafe":penalty_curtailment_unsafe,
#                         "rho_safe":rho_safe,
#                         "rho_danger":0.97,
#                         "margin_th_limit":0.93,
#                         "alpha_por_error":0.5}
#                         for penalty_storage_unsafe in [0.04, 0.07]
#                         for penalty_curtailment_unsafe in [0.04, 0.07]
#                         for rho_safe in [0.6, 0.9]
#                     ]

parameters_to_test = [{"penalty_redispatching_unsafe":10, 
                        "penalty_storage_unsafe": penalty_storage_unsafe, 
                        "penalty_curtailment_unsafe":penalty_curtailment_unsafe,
                        "rho_safe":rho_safe,
                        "rho_danger":0.97,
                        "margin_th_limit":0.93,
                        "alpha_por_error":0.5}
                        for penalty_storage_unsafe in [0.04]
                        for penalty_curtailment_unsafe in [0.04]
                        for rho_safe in [0.7, 0.75, 0.8]
                    ]

agents_dict = {}
for i, params in enumerate(parameters_to_test) :
    agents_dict.update({f"optim_{i}": OptimCVXPY(env.action_space,
                                                env,
                                                **params
                                                )}
                        )
    # parameters_to_test[i].update({"datetime_now":datetime_now, "agent_id":i})
    # with open("./pre_train/dicts_optimizers_params.json", 'a') as fp:
    #     json.dump(parameters_to_test[i], fp, indent=4)

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

        parameters_to_test[i].update({"datetime_now":datetime_now, "agent_name":agent_name, "agent_id":i})
        with open("./pre_train/total_results/dicts_optimizers_params.json", 'a') as fp:
            json.dump(parameters_to_test[i], fp, indent=4)
    except Exception as e:
        print(e)
        

with open('./pre_train/total_results/total_results_{}.npy'.format(datetime_now), 'wb') as f:
    np.save(f, total_results)