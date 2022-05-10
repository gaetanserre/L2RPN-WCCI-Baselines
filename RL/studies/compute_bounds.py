import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Parameters import Parameters
from grid2op.Runner import Runner
from grid2op.Agent import DoNothingAgent
from grid2op.Episode import EpisodeData
from grid2op.gym_compat import BoxGymObsSpace, GymEnv

import numpy as np

import os
import shutil
import json
import argparse

# Arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env_name",    default="l2rpn_wcci_2022_dev", type=str)
parser.add_argument("--attributes",  default=["load_p", "p_or"], nargs='+', type=list)
parser.add_argument("--seed",        default=42, type=int)
parser.add_argument("--nb_chronics", default=20, type=int)
parser.add_argument("--path_agents", default="output_agents_no_overflow", type=str)
parser.add_argument("--output",      default="data.json", type=str)
args = parser.parse_args()

env_name    = args.env_name
attributes  = args.attributes
seed        = args.seed
nb_chronics = args.nb_chronics
path_agents = args.path_agents
output      = args.output

# Create nb_chronics episode to sample the max and min value of each attribute
def create_episodes(env):
  np.random.seed(seed)
  chronics_to_use = np.random.choice(env.chronics_handler.real_data.available_chronics(), nb_chronics, replace=False)

  env.chronics_handler.real_data.set_filter(lambda x: x in chronics_to_use)
  env.chronics_handler.real_data.reset()


  os.makedirs(path_agents, exist_ok=True)

  for agentClass, agentName in zip([DoNothingAgent], ["DoNothingAgent"]):
    env.seed(0)
    path_this_agent = os.path.join(path_agents, agentName)
    shutil.rmtree(os.path.abspath(path_this_agent), ignore_errors=True)
    runner = Runner(**env.get_params_for_runner(),
                    agentClass=agentClass
                    )
    runner.run(path_save=path_this_agent, nb_episode=nb_chronics)

# Compute the max and min value of each attribute using the sample chronics
# previously computed
def get_attributes_data(env):
  data = {}
  for attr in attributes:
    env_gym = GymEnv(env)
    env_gym.observation_space = BoxGymObsSpace(env.observation_space, attr_to_keep=[attr])
    data[attr] = {"env": env_gym, "observations": []}


  for path, chronics in EpisodeData.list_episode(os.path.join(path_agents, "DoNothingAgent")):
    print(chronics)
    this_episode = EpisodeData.from_disk(path, chronics)
    for obs in this_episode.observations:
      for attr in attributes:
        data[attr]["observations"].append(data[attr]["env"].observation_space.to_gym(obs))

  for attr in attributes:
    env_gym = GymEnv(env)
    env_gym.observation_space = BoxGymObsSpace(env.observation_space, attr_to_keep=attributes)
    data[attr]["observations"] = np.array(data[attr]["observations"])
    data[attr]["min"] = np.min(data[attr]["observations"])
    data[attr]["max"] = np.max(data[attr]["observations"])

  return data

if __name__ == "__main__":
  p = Parameters()
  # Disable overflows to have th maximum number of observations
  p.NO_OVERFLOW_DISCONNECTION = True
  env = grid2op.make("l2rpn_wcci_2022_dev", param=p, backend=LightSimBackend())

  create_episodes(env)
  data = get_attributes_data(env)

  bounds = {}
  for attr in attributes:
    bounds[attr] = {"min": str(data[attr]["min"]), "max": str(data[attr]["max"])}

  with open(output, 'w') as fp:
    json.dump(bounds, fp)
