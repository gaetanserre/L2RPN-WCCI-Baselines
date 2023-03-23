import grid2op
from l2rpn_baselines.utils import GymEnvWithRecoWithDN
from grid2op.Observation import BaseObservation
from typing import Tuple, Dict
import numpy as np

class GymEnvWithSetPointNoEnv(GymEnvWithRecoWithDN): 
    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, ind=2, **kwargs):
        super().__init__(env_init=env_init, *args, reward_cumul=reward_cumul, safe_max_rho=safe_max_rho, **kwargs)
        self.ind = ind
        self._last_obs = None
        self.nb_time_step = None
        self.gymobs = None
        self.max_episode_duration = self.init_env.max_episode_duration()

    def reset(self, seed=None, return_info=False, options=None):
        # param = self.init_env.parameters
        # # param.INIT_STORAGE_CAPACITY = self.init_env.space_prng.uniform(size=self.init_env.n_storage)
        # param.INIT_STORAGE_CAPACITY = self.init_env.space_prng.uniform()
        # self.init_env.change_parameters(param)
        if seed is not None:
            self.seed(seed)
        self.gymobs = np.array([0.5, 0.5, 0.5, 0.5]) # commenter
        self.nb_time_step = 0
        self.storage_setpoint = np.clip(0.5+np.cumsum(self.init_env.space_prng.uniform(-0.05, 0.05, (self.init_env.max_episode_duration()+1, self.init_env.n_storage)), axis=0), 0,1)
        self._update_obs_attribute(self.gymobs, "storage_setpoint", self.storage_setpoint[self.nb_time_step, :])
        self._last_obs = self.gymobs
        return self.gymobs

    def _update_obs_attribute(self, gym_obs, attr_to_update, value):
        if attr_to_update == "storage_charge":
            gym_obs[0:2] = value
        elif attr_to_update == "storage_setpoint":
            gym_obs[2:4] = value
        else:
            raise RuntimeError(f"Unknown attribute {attr_to_update}")

    def _get_attr_gymobs(self, gym_obs, attr):
        # dims = [0] + self.observation_space._dims
        # for i, attr_tmp in enumerate(self.observation_space._attr_to_keep):
        #    if attr_tmp == attr:
        #        return gym_obs[dims[i]:dims[i+1]]
        if attr == "storage_charge":
            return gym_obs[0:2]
        elif attr == "storage_setpoint": 
            return gym_obs[2:4]
        else:
            raise RuntimeError(f"Unknown attribute {attr}")

    def _get_reward(self, gym_obs):
        if self.nb_time_step == 0:
            raise ValueError("You are still at timestep 0, hence there is no previous setpoint to update the reward")
        else :
            Emin, Emax = self.init_env.storage_Emin, self.init_env.storage_Emax
            storage_charge = self._get_attr_gymobs(gym_obs, "storage_charge") * (Emax - Emin) + Emin
            storage_setpoint_mwh = self.storage_setpoint[self.nb_time_step - 1, :] * (Emax - Emin) + Emin
            ind = self.ind
            k = 1/(0.5)**ind
            # print(storage_charge, storage_setpoint_mwh, k)
            reward = 1 - k * np.mean(np.power(np.abs(storage_setpoint_mwh - storage_charge)/(Emax-Emin), ind))
        return reward


    def get_obs(self):
        return self._last_obs.copy()

    def step(self, gym_action):
        # Get grid2op action
        gym_act_set_storage = gym_action * self.action_space._multiply["set_storage"] + self.action_space._add["set_storage"]
        # Update gym_obs
        gym_obs = self._last_obs
        self._update_obs_attribute(gym_obs, "storage_setpoint", self.storage_setpoint[self.nb_time_step, :])
        delta_obs_charge_norm = (gym_act_set_storage/12)/ self.observation_space._divide['storage_charge']
        self._update_obs_attribute(gym_obs, "storage_charge", np.clip(self._get_attr_gymobs(gym_obs, "storage_charge") + delta_obs_charge_norm, 0, 1))
        self.nb_time_step += 1
        # Update reward, done, info
        done = self.nb_time_step >= self.max_episode_duration
        info = {}
        reward = self._get_reward(gym_obs)
        # self._last_obs = gym_obs
        return gym_obs, float(reward), done, info