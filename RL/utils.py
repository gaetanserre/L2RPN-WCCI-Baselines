from inspect import Parameter
import numpy as np
import grid2op
from collections.abc import Iterable
from l2rpn_baselines.PPO_SB3 import train
from grid2op.Agent import RecoPowerlineAgent
from grid2op.utils import EpisodeStatistics
from grid2op.dtypes import dt_int
from lightsim2grid import LightSimBackend
from grid2op.Chronics import MultifolderWithCache
import json
import os
from grid2op.Parameters import Parameters
from grid2op.Reward import BaseReward

# import sys
# sys.path.insert(0, "examples/")

# from ppo_stable_baselines.A_prep_env import get_env_seed
# from ppo_stable_baselines.C_evaluate_trained_model import get_ts_survived_dn, get_ts_survived_reco, load_agent

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
      
      if "_val" in nm_:
        # save the normalization parameters from the validation set
        dict_ = {"subtract": {}, 'divide': {}}
        for attr_nm in ["gen_p", "load_p", "p_or", "rho"]:
          avg_ = stats_reco.get(attr_nm)[0].mean(axis=0)
          std_ = stats_reco.get(attr_nm)[0].std(axis=0)
          dict_["subtract"][attr_nm] = [float(el) for el in avg_]
          dict_["divide"][attr_nm] = [max(float(el), 1.0) for el in std_]

        with open("./preprocess_obs.json", "w", encoding="utf-8") as f:
          json.dump(obj=dict_, fp=f)

        act_space_kwargs = {"add": {"redispatch": [0. for gen_id in range(env_tmp.n_gen) if env_tmp.gen_redispatchable[gen_id]],
                                    "set_storage": [0. for _ in range(env_tmp.n_storage)]},
                            'multiply': {"redispatch": [1. / (max(float(el), 1.0)) for gen_id, el in enumerate(env_tmp.gen_max_ramp_up) if env_tmp.gen_redispatchable[gen_id]],
                                          "set_storage": [1. / (max(float(el), 1.0)) for el in env_tmp.storage_max_p_prod]}
                            }
        with open("./preprocess_act.json", "w", encoding="utf-8") as f:
          json.dump(obj=act_space_kwargs, fp=f)


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
  dict_to_json["learning_rate"] = dict_to_json["learning_rate"] if isinstance(dict_to_json["learning_rate"], float) else dict_to_json["learning_rate"].__name__
  dict_to_json["device"] = str(dict_to_json["device"])
  dict_to_json["reward"] = str(type(env.get_reward_instance()))
  os.makedirs(os.path.join(train_args["save_path"], train_args["name"]), exist_ok=True)
  with open(full_path, 'x') as fp:
    json.dump(dict_to_json, fp, indent=4)

  print("environment loaded !")

  with open("./preprocess_obs.json", "r", encoding="utf-8") as f:
    train_args["obs_space_kwargs"] = json.load(f)
  with open("./preprocess_act.json", "r", encoding="utf-8") as f:
    train_args["act_space_kwargs"] = json.load(f)

  return train(env, **train_args)


def iter_hyperparameters(env,
                         train_args:dict,
                         name:str,
                         hyperparam_name:str,
                         hyperparam_values: Iterable,
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

  return ts_survived, dn_ts_survived, reco_ts_survived


class CustomReward2(BaseReward):
    def __init__(self):
        """
        Initializes :attr:`BaseReward.reward_min` and :attr:`BaseReward.reward_max`

        """
        self.reward_min = 0
        self.reward_max = 1.
        self._min_rho = 0.90
        self._max_rho = 2.0
        
        # parameters init with the environment
        self._max_redisp = None
        self._1_max_redisp = None
        self._is_renew_ = None
        self._1_max_redisp_act = None
        self._nb_renew = None
    
    def initialize(self, env):
        self._max_redisp = np.maximum(env.gen_pmax - env.gen_pmin, 0.)
        self._max_redisp += 1
        self._1_max_redisp = 1.0 / self._max_redisp / env.n_gen
        self._is_renew_ = env.gen_renewable
        self._1_max_redisp_act = np.maximum(np.maximum(env.gen_max_ramp_up, env.gen_max_ramp_down), 1.0)
        self._1_max_redisp_act = 1.0 / self._1_max_redisp_act / np.sum(env.gen_redispatchable)
        self._nb_renew = np.sum(self._is_renew_)
        
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            res = np.sqrt(env.nb_time_step / env.max_episode_duration())
            print(f"{os.path.split(env.chronics_handler.get_id())[-1]}: {env.nb_time_step = }, reward : {res:.3f}")
            if env.nb_time_step <= 5:
                print(f"reason game over: {env.infos['exception']}")
            # episode is over => 2 cases
            # if env.nb_time_step == env.max_episode_duration():
            #     return self.reward_max
            # else:
            #     return self.reward_min
            return res
        
        if is_illegal or is_ambiguous or has_error:
            return self.reward_min
        # penalize the dispatch
        obs = env.get_obs()
        score_redisp_state = 0.
        # score_redisp_state = np.sum(np.abs(obs.target_dispatch) * self._1_max_redisp)
        # score_redisp_action = np.sum(np.abs(action.redispatch) * self._1_max_redisp_act) 
        # score_redisp = 0.5 *(score_redisp_state + score_redisp_action)
        
        # penalize the curtailment
        # score_curtail_state = 0.
        # score_curtail_state = np.sum(obs.curtailment_mw * self._1_max_redisp)
        # score_curtail_state = np.sum(1-obs.curtailment_limit) / self._nb_renew 
        score_curtail_state = np.sum(np.square(1-obs.curtailment_limit)) / self._nb_renew
        # curt_act = action.curtail
        # score_curtail_action = np.sum(curt_act[curt_act != -1.0]) / self._nb_renew 
        # score_curtail = 0.5 * (score_curtail_state + score_curtail_action)
        
        # rate the actions
        # score_action = 0.5 * (np.sqrt(score_redisp) + np.sqrt(score_curtail))

        # penalize batteries far from the middle charge
        distance_to_middle_charge = np.sum(np.abs(obs.storage_charge - (obs.storage_Emin + obs.storage_Emax)/2))
        distance_storage_max = np.sum((obs.storage_Emax - obs.storage_Emin)/2)
        score_storage_state = distance_to_middle_charge/distance_storage_max
        score_storage_action = np.sum(np.abs(action.set_storage)) / np.sum(np.maximum(obs.storage_max_p_absorb, obs.storage_max_p_prod))
        score_storage = 0.5 * (score_storage_action + score_storage_state)

        # score the "state" of the grid
        # tmp_state = np.minimum(np.maximum(obs.rho, self._min_rho), self._max_rho)
        # tmp_state -= self._min_rho
        # tmp_state /= (self._max_rho - self._min_rho) * env.n_line
        # score_state = np.sqrt(np.sqrt(np.sum(tmp_state)))
        score_state = 0.

        # score close to goal
        # score_goal = 0.
        # score_goal = env.nb_time_step / env.max_episode_duration()
        # score_goal = 1.0
        score_goal = 0.1
        
        # score too much redisp
        # res = score_goal * (1.0 - 0.5 * (score_action + score_state)
        res = score_goal * (1.0 - 0.5 * (score_curtail_state + score_storage))
        return res