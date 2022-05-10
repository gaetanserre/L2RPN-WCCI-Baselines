from l2rpn_baselines.utils import GymEnvWithHeuristics
from typing import List
from grid2op.Action import BaseAction
import numpy as np

class CustomGymEnv(GymEnvWithHeuristics):
  """This environment is slightly more complex that the other one.
  
  It consists in 2 things:
  
  #. reconnecting the powerlines if possible
  #. doing nothing is the state of the grid is "safe" (for this class, the notion of "safety" is pretty simple: if all
      flows are bellow 90% (by default) of the thermal limit, then it is safe)
  
  If for a given step, non of these things is applicable, the underlying trained agent is asked to perform an action
  
  .. warning::
      When using this environment, we highly recommend to adapt the parameter `safe_max_rho` to suit your need.
      
      Sometimes, 90% of the thermal limit is too high, sometimes it is too low.
      
  """
  def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, **kwargs):
    super().__init__(env_init, reward_cumul=reward_cumul, *args, **kwargs)
    self._safe_max_rho = safe_max_rho
    self.dn = self.init_env.action_space({})
        
  def heuristic_actions(self, g2op_obs, reward, done, info) -> List[BaseAction]:
    """To match the description of the environment, this heuristic will:
    
    - return the list of all the powerlines that can be reconnected if any
    - return the list "[do nothing]" is the grid is safe
    - return the empty list (signaling the agent should take control over the heuristics) otherwise

    Parameters
    ----------
    See parameters of :func:`GymEnvWithHeuristics.heuristic_actions`

    Returns
    -------
    See return values of :func:`GymEnvWithHeuristics.heuristic_actions`
    """
    
    to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)
    res = []
    if np.any(to_reco):
      # reconnect something if it can be
      reco_id = np.where(to_reco)[0]
      for line_id in reco_id:
          g2op_act = self.init_env.action_space({"set_line_status": [(line_id, +1)]})
          res.append(g2op_act)
    elif g2op_obs.rho.max() <= self._safe_max_rho:
      # play do nothing if there is "no problem" according to the "rule of thumb"
      res = [self.init_env.action_space()]
    return res

  def step(self, gym_action):
    """This function implements the special case of the "step" function (as seen by the "gym environment") that might
    call multiple times the "step" function of the underlying "grid2op environment" depending on the
    heuristic.

    It takes a gym action, convert it to a grid2op action (thanks to the action space) and
    simulates if this action is better than doing nothing. If so, it performs the action otherwise
    it performs the "do nothing" action.

    Then process the heuristics / expert rules / forced actions / etc. and return the next gym observation that will
    be processed by the agent.

    The number of "grid2op steps" can vary between different "gym environment" call to "step".

    It has the same signature as the `gym.Env` "step" function, of course. 

    Parameters
    ----------
    gym_action :
        the action (represented as a gym one) that the agent wants to perform.

    Returns
    -------
    gym_obs:
        The gym observation that will be processed by the agent
        
    reward: ``float``
        The reward of the agent (that might be computed by the )
        
    done: ``bool``
        Whether the episode is over or not
        
    info: Dict
        Other type of informations
        
    """
    g2op_act = self.action_space.from_gym(gym_action)

    _, sim_reward_act, _, _ = self.init_env.simulate(g2op_act)
    _, sim_reward_dn, _, _ = self.init_env.simulate(self.dn)
    if sim_reward_dn > sim_reward_act:
      g2op_act = self.dn

    g2op_obs, reward, done, info = self.init_env.step(g2op_act)
    if not done:
      g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, done, info)
    gym_obs = self.observation_space.to_gym(g2op_obs)
    return gym_obs, float(reward), done, info