import grid2op
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from grid2op.Observation import BaseObservation
from typing import Tuple, Dict
import numpy as np

class GymEnvWithSetPoint(GymEnvWithRecoWithDN): 
    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, alpha=1, **kwargs):
        super().__init__(env_init=env_init, *args, reward_cumul=reward_cumul, safe_max_rho=safe_max_rho, **kwargs)
        self.alpha = alpha
        self._last_obs = None

    def reset(self, seed=None, return_info=False, options=None):
        # param = self.init_env.parameters
        # # param.INIT_STORAGE_CAPACITY = self.init_env.space_prng.uniform(size=self.init_env.n_storage)
        # param.INIT_STORAGE_CAPACITY = self.init_env.space_prng.uniform()
        # self.init_env.change_parameters(param)
        gymobs_tmp = super().reset(seed=seed, return_info=return_info, options=options)
        self.storage_setpoint = np.clip(0.5+np.cumsum(self.init_env.space_prng.uniform(-0.05, 0.05, (self.init_env.max_episode_duration()+1, self.init_env.n_storage)), axis=0), 0,1)
        gymobs = self._update_setpoint(gymobs_tmp, "storage_capacity_setpoint", self.storage_setpoint[self.init_env.nb_time_step, :])
        self._last_obs = gymobs
        return gymobs

    def _update_setpoint(self, gym_obs, attr_to_update, setpoint):
        dims = [0] + self.observation_space._dims
        for i, attr_tmp in enumerate(self.observation_space._attr_to_keep):
            if attr_tmp == attr_to_update:
                gym_obs[dims[i]:dims[i+1]]= setpoint
        return gym_obs

    def _update_reward(self, g2op_obs, reward):
        if self.init_env.nb_time_step == 0:
            raise ValueError("You are still at timestep 0, hence there is no previous setpoint to update the reward")
        else :
            Emin = self.init_env.storage_Emin
            Emax = self.init_env.storage_Emax
            reward -= self.alpha * np.sum(((g2op_obs.storage_charge-Emin)/(Emax-Emin) - self.storage_setpoint[self.init_env.nb_time_step - 1, :])**2) / self.init_env.n_storage
            rew_min, rew_max = - self.alpha, 1
            reward = (reward - rew_min)/(rew_max - rew_min)

        return reward

    def apply_heuristics_actions(self,
                                 g2op_obs: BaseObservation,
                                 reward: float,
                                 done: bool,
                                 info: Dict ) -> Tuple[BaseObservation, float, bool, Dict]:
        need_action = True
        res_reward = reward
        
        tmp_reward = reward
        tmp_info = info
        while need_action:
            need_action = False
            g2op_actions = self.heuristic_actions(g2op_obs, tmp_reward, done, tmp_info)
            for g2op_act in g2op_actions:
                need_action = True
                tmp_obs, tmp_tmp_reward, tmp_done, tmp_info = self.init_env.step(g2op_act)
                self._last_obs = self.observation_space.to_gym(tmp_obs)
                tmp_reward = self._update_reward(tmp_obs, tmp_tmp_reward)

                g2op_obs = tmp_obs
                done = tmp_done
                
                if self._reward_cumul == "max":
                    res_reward = max(tmp_reward, res_reward)
                elif self._reward_cumul == "sum":
                    res_reward += tmp_reward
                elif self._reward_cumul == "last":
                    res_reward = tmp_reward
                    
                if tmp_done:
                    break
            if done:
                break
        return g2op_obs, res_reward, done, info

    def get_obs(self):
        return self._last_obs.copy()

    def step(self, gym_action):
        g2op_act_tmp = self.action_space.from_gym(gym_action)
        g2op_act = self.fix_action(g2op_act_tmp)
        g2op_obs, reward_tmp, done, info = self.init_env.step(g2op_act)
        reward = self._update_reward(g2op_obs, reward_tmp)
        if not done:
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, done, info)
        gym_obs_tmp = self.observation_space.to_gym(g2op_obs)
        gym_obs = self._update_setpoint(gym_obs_tmp, "storage_capacity_setpoint", self.storage_setpoint[self.init_env.nb_time_step, :])
        self._last_obs = gym_obs
        return gym_obs, float(reward), done, info