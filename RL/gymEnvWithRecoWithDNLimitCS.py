from l2rpn_baselines.utils import GymEnvWithHeuristics
from grid2op.Chronics.multiFolder import Multifolder
import numpy as np

# TODO GymEnvWithHeuristics does not work because of a bug in l2rpn-baselines...
# see https://github.com/rte-france/l2rpn-baselines/issues/43

class GymEnvWithHeuristicsWithDNFixed(GymEnvWithHeuristics):
    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, **kwargs):
        super().__init__(env_init, *args, **kwargs)
        self.prev_obs = None
        self._safe_max_rho = safe_max_rho
        
    def fix_action(self, obs, grid2op_action):
        """This function can be used to "fix" / "modify" / "cut" / "change"
        a grid2op action just before it will be applied to the underlying "env.step(...)"
        
        This can be used, for example to "limit the curtailment or storage" of the
        action in case this one is too strong and would lead to a game over.

        By default it does nothing.
        
        Parameters
        ----------
        grid2op_action : _type_
            _description_
            
        """
        return grid2op_action

    def step(self, gym_action):
        """
        This is a copy pasted with the 
        """
        g2op_act_tmp = self.action_space.from_gym(gym_action)
        g2op_act = self.fix_action(self.prev_obs, g2op_act_tmp)
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        if not done:
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, done, info)
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, float(reward), done, info
    
    def heuristic_actions(self, g2op_obs, reward, done, info):
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
    
    def reset(self, seed=None, return_info=False, options=None):
        """This function implements the "reset" function. It is called at the end of every episode and
        marks the beginning of a new one.
        
        Again, before the agents sees any observations from the environment, they are processed by the 
        "heuristics" / "expert rules".
        
        .. note::
            The first observation seen by the agent is not necessarily the first observation of the grid2op environment.

        Returns
        -------
        gym_obs:
            The first open ai gym observation received by the agent
        """
        done = True
        info = {}  # no extra information provided !
        while done:
            super().reset(seed, return_info, options)  # reset the scenario
            g2op_obs = self.init_env.get_obs()  # retrieve the observation
            reward = self.init_env.reward_range[0]  # the reward at first step is always minimal
            
            # perform the "heuristics" steps
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, False, info)
            self.prev_obs = g2op_obs
            # convert back the observation to gym
            gym_obs = self.observation_space.to_gym(g2op_obs)
            
        if return_info:
            return gym_obs, info
        else:
            return gym_obs
                   
class GymEnvWithRecoWithDNWithCS(GymEnvWithHeuristicsWithDNFixed):    
    """this env shuffles the chronics order from time to time. We use it at training time !"""
    def __init__(self,
                 env_init,
                 *args,
                 reward_cumul="init",
                 safe_max_rho=0.9,
                 cs_margin=150,
                 **kwargs):
        super().__init__(env_init, *args, reward_cumul=reward_cumul, safe_max_rho=safe_max_rho, **kwargs)
        self.cs_margin = cs_margin
    
    def fix_action(self, grid2op_obs, grid2op_action):
        grid2op_action.limit_curtail_storage(grid2op_obs, margin=self.cs_margin)
        return grid2op_action
