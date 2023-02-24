import grid2op
from l2rpn_baselines.utils import GymEnvWithRecoWithDN

class GymEnvStorage(GymEnvWithRecoWithDN): 
    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, **kwargs):
        super().__init__(env_init=env_init, *args, reward_cumul=reward_cumul, safe_max_rho=safe_max_rho, **kwargs)
        self.reset = self.new_reset

    def new_reset(self, seed=None, return_info=False, options=None):
        param = self.init_env.parameters
        param.INIT_STORAGE_CAPACITY = self.init_env.space_prng.uniform()
        self.init_env.change_parameters(param)
        return super().reset(seed=seed, return_info=return_info, options=options)
