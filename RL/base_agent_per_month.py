from grid2op.Agent import BaseAgent
import numpy as np
import json
import os
import re
from l2rpn_baselines.PPO_SB3 import evaluate
from GymEnvWithRecoWithDNWithShuffle import GymEnvWithRecoWithDNWithShuffle


class BaseAgentPerMonth(BaseAgent):
    def __init__(self, l2rpn_agents_list, limit_cs_margin):
        BaseAgent.__init__(self, l2rpn_agents_list[0].action_space)
        self.l2rpn_agents_list = l2rpn_agents_list
        self.limit_cs_margin = limit_cs_margin
    
    def act(self, obs, reward, done=False):
        l2rpn_agent = self.l2rpn_agents_list[obs.month-1]
        action = l2rpn_agent.act(obs, reward, done)
        # We try to limit to end up with a "game over" because actions on curtailment or storage units.
        action.limit_curtail_storage(obs, margin=self.limit_cs_margin)
        return action


def make_agent_per_month(env, submission_dir, agents_dir, limit_cs_margin=60, gymenv_kwargs={"safe_max_rho": 0.99}):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your submission directory and return a valid agent.
    """

    # normalization_dir = os.path.join(submission_dir, "normalization")
    normalization_dir = submission_dir
    agents_dir        = os.path.join(submission_dir, agents_dir)

    with open(os.path.join(normalization_dir, "preprocess_obs.json"), 'r', encoding="utf-8") as f:
      obs_space_kwargs = json.load(f)
    with open(os.path.join(normalization_dir, "preprocess_act.json"), 'r', encoding="utf-8") as f:
      act_space_kwargs = json.load(f)


    agents = []
    patterns = list(map(lambda x: re.compile(".*month"+ x + ".*"), ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]))
    for i_month in range(12):
        pattern = patterns[i_month]
        agent_name = list(filter(pattern.match, os.listdir(agents_dir)))[0] # 1st agent name matching the month
        l2rpn_agent, _ = evaluate(env,
                      nb_episode=0,
                      load_path=agents_dir,
                      name=agent_name,
                      gymenv_class=GymEnvWithRecoWithDNWithShuffle,
                      gymenv_kwargs=gymenv_kwargs,
                      obs_space_kwargs=obs_space_kwargs,
                      act_space_kwargs=act_space_kwargs)
        agents.append(l2rpn_agent)
  
    return BaseAgentPerMonth(agents, limit_cs_margin)


if __name__ == "__main__":
  from lightsim2grid import LightSimBackend
  import grid2op

  env = grid2op.make("/home/boguslawskieva/L2RPN-WCCI-Baselines/input_data_val", backend=LightSimBackend())
  agent_set = make_agent_per_month(env, ".", "saved_model/expe_train_per_month/GymEnvWithRecoDNShuffle_per_month_0") 

  nb_steps = 0
  obs = env.reset()
  done = False
  reward = 0
  while not done:
    nb_steps += 1
    action = agent_set.act(obs, reward, done)
    print(action)
    obs, reward, done, _ = env.step(action)
  
  print(f"Nb steps: {nb_steps}")