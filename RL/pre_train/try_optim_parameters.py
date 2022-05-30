import sys
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

os.chdir('/home/boguslawskieva/L2RPN-WCCI-Baselines/RL')

save_path = "./saved_model/"
ENV_NAME = "l2rpn_wcci_2022_dev"
SCOREUSED = ScoreL2RPN2020
verbose = False
p = Parameters()
p.LIMIT_INFEASIBLE_CURTAILMENT_STORAGE_ACTION = True
datetime_now=datetime.now().strftime('%Y-%m-%d_%H-%M')
  
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

parameters_to_test = [{"penalty_redispatching_unsafe":0, 
                        "penalty_storage_unsafe":0.01, 
                        "penalty_curtailment_unsafe":0.01,
                        "rho_safe":0.95,
                        "rho_danger":0.97,
                        "margin_th_limit":0.93,
                        "alpha_por_error":0.5},
                        {"penalty_redispatching_unsafe":1, 
                        "penalty_storage_unsafe":0.01, 
                        "penalty_curtailment_unsafe":0.01,
                        "rho_safe":0.95,
                        "rho_danger":0.97,
                        "margin_th_limit":0.93,
                        "alpha_por_error":0.5}
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
    results = eval_agent(ENV_NAME,
            nb_scenario,
            agent_name,
            save_path,
            SCOREUSED,
            verbose,
            gymenv_kwargs=train_args["gymenv_kwargs"],
            param=p,
            filter_fun=filter_chronics,
            my_agent=my_agent
            )
    for k in range(3):
        total_results[i, :, k]=np.array(results[k])

    parameters_to_test[i].update({"datetime_now":datetime_now, "agent_name":agent_name, "agent_id":i})
    with open("dicts_optimizers_params.json", 'a') as fp:
        json.dump(parameters_to_test[i], fp, indent=4)

with open('./pretrain/total_results_{}.npy'.format(datetime_now), 'wb') as f:
    np.save(f, total_results)