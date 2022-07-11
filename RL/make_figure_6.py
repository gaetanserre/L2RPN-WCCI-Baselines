import argparse
import json
import os
import numpy as np
from tqdm import tqdm
from lightsim2grid import LightSimBackend

import grid2op
from grid2op.Agent import BaseAgent
from grid2op.utils import ScoreL2RPN2022
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace
    
from l2rpn_baselines.PPO_SB3.utils import SB3Agent

from GymEnvWithRecoWithDNWithShuffle import GymEnvWithRecoWithDNWithShuffle

from agent_experiments import ENV_NAME, check_cuda


def cli():
    parser = argparse.ArgumentParser(description="Train baseline PPO")
    parser.add_argument("--has_cuda", default=1, type=int,
                        help="Is pytorch installed with cuda support ? (default True)")
    
    parser.add_argument("--cuda_device", default=0, type=int,
                        help="Which cuda device to use for pytorch (only used if you want to use cuda)")
    
    parser.add_argument("--safe_max_rho", default=0.9, type=float,
                        help="safe_max_rho to use for evaluation (default 0.9)")
    
    parser.add_argument("--training_iter", default=10_000_000, type=int,
                        help="At which training iteration you want to load the agents ?")
    
    parser.add_argument("--agent_name", default="GymEnvWithRecoWithDN", type=str,
                        help="Name for your agent, default 'GymEnvWithRecoWithDN'")
    
    parser.add_argument("--path_test_set",
                        default="../input_data_val",
                        type=str,
                        help="Name of the environment on which you want to evaluate your agents")
    
    parser.add_argument("--path_config_set",
                        default="../ingestion_program_val",
                        type=str,
                        help="Name of configuration file for the environment (mainly the seeds)")
    
    parser.add_argument("--path_agents", default="../../2022_ADPRL_paper/",
                        type=str,
                        help=("Path where the agents are stored. They are in the format `path_agents/A_DIRECTORY/AGENTS` "
                              "(ie in a sub directory of the path_agents argument)"))
    
    parser.add_argument("--limit_cs_margin", default=150.,
                        type=float,
                        help=("The margin used in `action.limit_curtail_storage(..., margin=XXX)` in the agent"))
    
    return parser.parse_args()


class BaselineAgent(BaseAgent):
    """see https://github.com/rte-france/l2rpn-baselines/issues/43
    a feature is missing in the l2rpn baselines repo"""
    def __init__(self, l2rpn_agent, limit_cs_margin):
        BaseAgent.__init__(self, l2rpn_agent.action_space)
        self.l2rpn_agent = l2rpn_agent
        self.limit_cs_margin = limit_cs_margin
    
    def act(self, obs, reward, done=False):
        action = self.l2rpn_agent.act(obs, reward, done)
        # We try to limit to end up with a "game over" because actions on curtailment or storage units.
        action.limit_curtail_storage(obs, margin=self.limit_cs_margin)
        return action


def get_agent(submission_dir, agent_dir, weights_dir, env, safe_max_rho, limit_cs_margin):
    """this is basically a copy paste of the PPO_SB3 evaluate function with some minor modification
    used to load the correct weights
    """
    
    # compute the score of said agent
    with open(os.path.join(submission_dir, "preprocess_obs.json"), 'r', encoding="utf-8") as f:
        obs_space_kwargs = json.load(f)
    with open(os.path.join(submission_dir, "preprocess_act.json"), 'r', encoding="utf-8") as f:
        act_space_kwargs = json.load(f)
    

    # load the attributes kept
    with open(os.path.join(agent_dir, "obs_attr_to_keep.json"), encoding="utf-8", mode="r") as f:
        obs_attr_to_keep = json.load(fp=f)
    with open(os.path.join(agent_dir, "act_attr_to_keep.json"), encoding="utf-8", mode="r") as f:
        act_attr_to_keep = json.load(fp=f)

    # create the action and observation space
    gym_observation_space =  BoxGymObsSpace(env.observation_space,
                                            attr_to_keep=obs_attr_to_keep,
                                            **obs_space_kwargs)
    gym_action_space = BoxGymActSpace(env.action_space,
                                      attr_to_keep=act_attr_to_keep,
                                      **act_space_kwargs)
    
    if os.path.exists(os.path.join(agent_dir, ".normalize_act")):
        for attr_nm in act_attr_to_keep:
            if (("multiply" in act_space_kwargs and attr_nm in act_space_kwargs["multiply"]) or 
                ("add" in act_space_kwargs and attr_nm in act_space_kwargs["add"]) 
               ):
                continue
            gym_action_space.normalize_attr(attr_nm)

    if os.path.exists(os.path.join(agent_dir, ".normalize_obs")):
        for attr_nm in obs_attr_to_keep:
            if (("divide" in obs_space_kwargs and attr_nm in obs_space_kwargs["divide"]) or 
                ("subtract" in obs_space_kwargs and attr_nm in obs_space_kwargs["subtract"]) 
               ):
                continue
            gym_observation_space.normalize_attr(attr_nm)
    
    # create the gym environment for the PPO agent...
    gymenv = GymEnvWithRecoWithDNWithShuffle(env, safe_max_rho=float(safe_max_rho))    
    gymenv.action_space.close()
    gymenv.action_space = gym_action_space
    gymenv.observation_space.close()
    gymenv.observation_space = gym_observation_space
    
    # create a grid2gop agent based on that (this will reload the save weights)
    l2rpn_agent = SB3Agent(env.action_space,
                           gym_action_space,
                           gym_observation_space,
                           nn_path=weights_dir,
                           gymenv=gymenv
                           )
    
    agent_to_evaluate = BaselineAgent(l2rpn_agent, limit_cs_margin)
    return agent_to_evaluate
        
if __name__ == "__main__":
    args = cli()
    use_cuda = check_cuda(args)

    # create the environment
    env = grid2op.make(str(args.path_test_set), backend=LightSimBackend())
    
    # read the seeds and other configuration
    config_file = os.path.join(os.path.abspath(args.path_config_set), "config_val.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    env_seeds =  [int(config["episodes_info"][os.path.split(el)[-1]]["seed"]) for el in sorted(env.chronics_handler.real_data.subpaths)]
    
    # initialize the class to compute the losses
    score_fun = ScoreL2RPN2022(env,
                               env_seeds=env_seeds,
                               nb_scenario=int(config["nb_scenario"]),
                               min_losses_ratio=float(config["score_config"]["min_losses_ratio"]),
                               max_step=-1,
                               nb_process_stats=1)
    
    safe_max_rho = float(args.safe_max_rho)
    training_iter = int(args.training_iter)
    limit_cs_margin = float(args.limit_cs_margin)
    res = {}
    
    safe_max_rho_str = f"{safe_max_rho:.4f}"
    res[safe_max_rho_str] = {}
    
    training_iter_str = f"{training_iter}"
    res[safe_max_rho_str][training_iter_str] = {}
    
    limit_cs_margin_str = f"{limit_cs_margin:.2f}"
    res[safe_max_rho_str][training_iter_str][limit_cs_margin_str] = {}
    # loop for all agents
    root_dir = os.path.abspath(args.path_agents)
    for machine_dir in tqdm(os.listdir(root_dir)):
        submission_dir  = os.path.join(root_dir, machine_dir)
        for name in tqdm(os.listdir(submission_dir)):
            agent_dir = os.path.join(submission_dir, name)
            weights_dir = os.path.join(agent_dir, f"{name}_{training_iter}_steps.zip")
            if os.path.exists(weights_dir):
                # I got a valid agent !
                agent_to_evaluate = get_agent(submission_dir, agent_dir, weights_dir, env, safe_max_rho)
                scores, n_played, total_ts = score_fun.get(agent_to_evaluate,
                                                           # path_save=path_save,  # TODO
                                                           nb_process=1)
                
                weights_dir_str = f"{weights_dir}"
                this_run = {"score_avg": float(np.mean(scores)),
                            "scores": [float(el) for el in scores], 
                            "score_std": float(np.std(scores)),
                            "min": float(np.min(scores)),
                            "max": float(np.max(scores)),
                            "n_played": [int(el) for el in n_played], 
                            "total_ts": [int(el) for el in total_ts]
                            }
                res[safe_max_rho_str][training_iter_str][limit_cs_margin_str][weights_dir_str] = this_run
                with open("agents_runs.json", "w", encoding="utf-8") as f:
                    json.dump(obj=res, fp=f)
