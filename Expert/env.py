import grid2op

class Env:
  def __init__(self, env_path=None, env_name=None, threshold=0.98):
    if env_path is None and env_name is None:
      raise ValueError("Please provide env_path or env_name")
    elif env_path is None:
      test = "educ" in env_name
      self.env = grid2op.make(env_name, test=test)
    else:
      raise NotImplementedError
    
    #self.env.reset()
    
    self.threshold = threshold
    self.do_nothing = self.env.action_space({})
  
  def reset(self):
    return self.env.reset()
  
  def step(self, action):
    action = self.env.action_space(action)
    obs, reward, done, info = self.env.step(action)

    while not done and obs.rho.max() < self.threshold:
      obs, reward, done, info = self.env.step(self.do_nothing)

    return obs, reward, done, info