import numpy as np
import grid2op
from typing import List
from l2rpn_baselines.PPO_SB3 import train
from grid2op.Agent import RecoPowerlineAgent
from grid2op.utils import EpisodeStatistics
from grid2op.dtypes import dt_int
from lightsim2grid import LightSimBackend
import json
import os
from grid2op.Parameters import Parameters

# Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from examples.ppo_stable_baselines.A_prep_env import get_env_seed
from examples.ppo_stable_baselines.C_evaluate_trained_model import get_ts_survived_dn, get_ts_survived_reco, load_agent


def split_train_val_test_sets(env, deep_copy):
  env.seed(1)
  env.reset()
  return env.train_val_split_random(add_for_test="test",
                                    pct_val=4.2,
                                    pct_test=4.2,
                                    deep_copy=deep_copy)

def generate_statistics(env_list, SCOREUSED, nb_process_stats, name_stats, verbose, filter_fun=None):
  # computes some statistics for environments in env_list (ex : val / test) to compare performance of 
  # some agents with the do nothing for example
  
  max_int = np.iinfo(dt_int).max
  for nm_ in env_list:
    env_tmp = grid2op.make(nm_, backend=LightSimBackend())
    if filter_fun is not None:
      env_tmp.chronics_handler.real_data.set_filter(filter_fun)
      env_tmp.chronics_handler.real_data.reset()
    is_statistics_already_computed = np.all([os.path.exists(os.path.join(grid2op.get_current_local_dir(), 
                        nm_, 
                        "_statistics_"+name_stats, 
                        os.path.basename(el))) 
                        for el in env_tmp.chronics_handler.real_data.available_chronics()])
    if not is_statistics_already_computed:
      nb_scenario = len(env_tmp.chronics_handler.subpaths)
      print(f"{nm_}: {nb_scenario}")
      my_score = SCOREUSED(env_tmp,
                          nb_scenario=nb_scenario,
                          env_seeds=np.random.randint(low=0,
                                                      high=max_int,
                                                      size=nb_scenario,
                                                      dtype=dt_int),
                          agent_seeds=[0 for _ in range(nb_scenario)],
                          verbose=verbose,
                          nb_process_stats=nb_process_stats)
      # compute statistics for reco powerline
      seeds = get_env_seed(nm_)
      reco_powerline_agent = RecoPowerlineAgent(env_tmp.action_space)
      stats_reco = EpisodeStatistics(env_tmp, name_stats=name_stats)
      stats_reco.compute(nb_scenario=nb_scenario,
                        agent=reco_powerline_agent,
                        env_seeds=seeds)


def train_agent(env, train_args:dict, max_iter:int = None):
  """
  This function trains an agent using the PPO algorithm
  with the arguments described in train_args.
  
  Parameters
  ----------

  env: :class:`grid2op.Environment`
      The environment on which you need to train your agent.

  train_args: `dict`
             A dictionnary of parameters for the train algorithm.
  
  max_iter: ``int``
           The number of iterations on which the environment
           will be restricted e.g 7 * 24 * 12 for a week.
           None to no restriction.

  Returns
  ----------

  baseline: 
        The trained baseline as a stable baselines PPO element.
  """

  if max_iter is not None:
    env.set_max_iter(max_iter)
  _ = env.reset()
  # env.chronics_handler.real_data.set_filter(lambda x: re.match(r".*february_000$", x) is not None)
  # env.chronics_handler.real_data.set_filter(lambda x: re.match(r".*00$", x) is not None)
  # see https://grid2op.readthedocs.io/en/latest/environment.html#optimize-the-data-pipeline
  # for more information !
  full_path = os.path.join(train_args["save_path"], train_args["name"], 'dict_train_args.json')
  dict_to_json = train_args.copy()
  dict_to_json["n_available_chronics"] = len(env.chronics_handler.real_data.available_chronics())
  dict_to_json["gymenv_class"] = dict_to_json["gymenv_class"].__name__
  dict_to_json["device"] = str(dict_to_json["device"])
  os.makedirs(os.path.join(train_args["save_path"], train_args["name"]), exist_ok=True)
  with open(full_path, 'x') as fp:
    json.dump(dict_to_json, fp, indent=4)

  print("environment loaded !")
  return train(env, **train_args)


def iter_hyperparameters(env,
                         train_args:dict,
                         name:str,
                         hyperparam_name:str,
                         hyperparam_values: List,
                         max_iter:int = None):
  """
  For each value v contained in `hyperparam_values`, this function
  trains an agent by setting the hyperparameter `hyperparam_name` to v.
  
  Parameters
  ----------

  env: :class:`grid2op.Environment`
      The environment on which you need to train your agent.

  train_args: `dict`
             A dictionnary of parameters for the train algorithm.
    
  name: `str`
        The initial name of the agent.

  hyperparam_name: `str`
                   The name of the hyperparameter to modify.
  
  hyperparam_values: `Iterable`
                    The values of the hyperparameter to modify.
  
  max_iter: `int`
           The number of iterations on which the environment
           will be restricted. e.g: 7 * 24 * 12 for a week.
           None to no restriction.

  Returns
  ----------

  baseline: `list`
        The list of the trained agents along with their names.
  """
  ret_agents = []

  for i, v in enumerate(hyperparam_values):
    train_args["name"] = '_'.join([name, hyperparam_name, str(i)])
    train_args[hyperparam_name] = v

    ret_agents.append((train_args["name"], train_agent(env, train_args, max_iter)))
  
  return ret_agents


def eval_agent(env_name: str,
               nb_scenario: int,
               agent_name: str,
               load_path: str,
               SCOREUSED,
               gymenv_class,
               verbose,
               nb_process_stats=1,
               gymenv_kwargs={},
               param=Parameters(),
               filter_fun=None):
  """
  This function evaluates a trained agent by comparing it to a DoNothing agent
  and a RecoPowerlineAgent.
  
  Parameters
  ----------

  env_name: `str`
      The environment name on which evaluate the agents.

  nb_scenario: `int`
              Number of scenarios to test the agents on.
  
  agent_name: `str`
           The name of the agent.
  
  load_path: `str`
           The path where the trained agent is stored.

  Returns
  ----------

  baseline: `list`
          The list of the steps survived by each agent.
        
  """

  # create the environment
  env_val = grid2op.make(env_name, backend=LightSimBackend(), param=param)
  if filter_fun is not None:
    env_val.chronics_handler.real_data.set_filter(filter_fun)
    env_val.chronics_handler.real_data.reset()
  
  # retrieve the reference data
  dn_ts_survived = get_ts_survived_dn(env_name, nb_scenario)
  reco_ts_survived = get_ts_survived_reco(env_name, nb_scenario)

  my_score = SCOREUSED(env_val,
                        nb_scenario=nb_scenario,
                        env_seeds=get_env_seed(env_name)[:nb_scenario],
                        agent_seeds=[0 for _ in range(nb_scenario)],
                        verbose=verbose,
                        nb_process_stats=nb_process_stats)

  my_agent = load_agent(env_val, load_path=load_path, name=agent_name, gymenv_class=gymenv_class, gymenv_kwargs=gymenv_kwargs)
  _, ts_survived, _ = my_score.get(my_agent)
  
  # compare with do nothing
  best_than_dn = 0
  for my_ts, dn_ts in zip(ts_survived, dn_ts_survived):
      print(f"\t{':-)' if my_ts >= dn_ts else ':-('} I survived {my_ts} steps vs {dn_ts} for do nothing ({my_ts - dn_ts})")
      best_than_dn += my_ts >= dn_ts
  print(f"The agent \"{agent_name}\" beats \"do nothing\" baseline in {best_than_dn} out of {len(dn_ts_survived)} episodes")
  
  # compare with reco powerline
  best_than_reco = 0
  for my_ts, reco_ts in zip(ts_survived, reco_ts_survived):
      print(f"\t{':-)' if my_ts >= reco_ts else ':-('} I survived {my_ts} steps vs {reco_ts} for reco powerline ({my_ts - reco_ts})")
      best_than_reco += my_ts >= reco_ts
  print(f"The agent \"{agent_name}\" beats \"reco powerline\" baseline in {best_than_reco} out of {len(reco_ts_survived)} episodes")

  return np.array(ts_survived), np.array(dn_ts_survived), np.array(reco_ts_survived)

def create_bar_plot(ts_survived, dn_ts_survived, reco_ts_survived, path=None):
  scenarios = [i+1 for i in range(len(ts_survived))] + ["Mean"]
  d = {"Scenarios": scenarios,
       "Agent": np.append(ts_survived, [ts_survived.mean()]),
       "Do Nothing": np.append(dn_ts_survived, dn_ts_survived.mean()),
       "Reco": np.append(reco_ts_survived, reco_ts_survived.mean())}
       
  df = pd.DataFrame(d)
  tidy = df.melt(id_vars="Scenarios").rename(columns=str.title)
  tidy.rename(columns={"Value": "Survived steps", "Variable": "Agents"}, inplace=True)
  fig, ax1 = plt.subplots(figsize=(10, 5))
  sns.barplot(x="Scenarios", y="Survived steps", hue="Agents", data=tidy, palette=("magma"), ax=ax1)
  ax1.axhline(2017, color="black", label="Maximum time steps")
  plt.legend()
  sns.despine(fig)
  if path is not None:
    plt.savefig(path)
  plt.show()