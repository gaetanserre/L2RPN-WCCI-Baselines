#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import argparse

from grid2op.MakeEnv import make
from grid2op.Parameters import Parameters

from l2rpn_baselines.SliceRDQN.sliceRDQN import SliceRDQN as RDQNAgent
from l2rpn_baselines.SliceRDQN.sliceRDQN_Config import SliceRDQN_Config as RDQNConfig

DEFAULT_NAME = "SliceRDQN"
DEFAULT_SAVE_DIR = "./models"
DEFAULT_LOG_DIR = "./logs-train"
DEFAULT_PRE_STEPS = 256
DEFAULT_TRAIN_STEPS = 1024 * 1024 * 10
DEFAULT_TRACE_LEN = 12
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-5
DEFAULT_VERBOSE = True

def cli():
    parser = argparse.ArgumentParser(description="Train baseline DDQN")

    # Paths
    parser.add_argument("--name", required=False, default="SliceRDQN_ls",
                        help="The name of the model")
    parser.add_argument("--data_dir", required=False,
                        default="l2rpn_case14_sandbox",
                        help="Path to the dataset root directory")
    parser.add_argument("--save_dir", required=False,
                        default=DEFAULT_SAVE_DIR, type=str,
                        help="Directory where to save the model")
    parser.add_argument("--load_file", required=False,
                        help="Path to model.h5 to resume training with")
    parser.add_argument("--logs_dir", required=False,
                        default=DEFAULT_LOG_DIR, type=str,
                        help="Directory to save the logs")
    # Params
    parser.add_argument("--num_pre_steps", required=False,
                        default=DEFAULT_PRE_STEPS, type=int,
                        help="Number of random steps before training")
    parser.add_argument("--num_train_steps", required=False,
                        default=DEFAULT_TRAIN_STEPS, type=int,
                        help="Number of training iterations")
    parser.add_argument("--trace_length", required=False,
                        default=DEFAULT_TRACE_LEN, type=int,
                        help="Number of stacked states to use during training")
    parser.add_argument("--batch_size", required=False,
                        default=DEFAULT_BATCH_SIZE, type=int,
                        help="Mini batch size (defaults to 1)")
    parser.add_argument("--learning_rate", required=False,
                        default=DEFAULT_LR, type=float,
                        help="Learning rate for the Adam optimizer")

    return parser.parse_args()


def train(env,
          name=DEFAULT_NAME,
          iterations=DEFAULT_TRAIN_STEPS,
          save_path=DEFAULT_SAVE_DIR,
          load_path=None,
          logs_path=DEFAULT_LOG_DIR,
          num_pre_training_steps=DEFAULT_PRE_STEPS,
          trace_length=DEFAULT_TRACE_LEN,
          batch_size=DEFAULT_BATCH_SIZE,
          learning_rate=DEFAULT_LR,
          verbose=DEFAULT_VERBOSE):

    """
    .. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
        
    """
    import tensorflow as tf
    # Set config
    RDQNConfig.LR = learning_rate
    RDQNConfig.BATCH_SIZE = batch_size
    RDQNConfig.TRACE_LENGTH = trace_length
    RDQNConfig.VERBOSE = verbose

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    agent = RDQNAgent(env.observation_space,
                      env.action_space,
                      name=name, 
                      is_training=True)

    if load_path is not None:
        agent.load(load_path)

    agent.train(env,
                iterations,
                save_path,
                num_pre_training_steps,
                logs_path)
    

if __name__ == "__main__":
    args = cli()

    # Set custom params
    param = Parameters()
    #param.NO_OVERFLOW_DISCONNECTION = True


    env = make(args.data_dir,
               param=param,
               action_class=TopologyAction,
               reward_class=CombinedScaledReward)

    # Do not load entires scenario at once
    # (faster exploration)
    env.set_chunk_size(100)

    # Register custom reward for training
    cr = env.reward_helper.template_reward
    #cr.addReward("bridge", BridgeReward(), 1.0)
    #cr.addReward("distance", DistanceReward(), 5.0)
    cr.addReward("overflow", CloseToOverflowReward(), 1.0)
    cr.addReward("game", GameplayReward(), 2.0)
    #cr.addReward("eco", EconomicReward(), 2.0)
    cr.addReward("reco", LinesReconnectedReward(), 1.0)
    cr.set_range(20.0, 30.0)
    # Initialize custom rewards
    cr.initialize(env)

    train(env,
          name = args.name,
          iterations = args.num_train_steps,
          num_pre_training_steps = args.num_pre_steps,
          save_path = args.save_dir,
          load_path = args.load_file,
          logs_path = args.logs_dir,
          trace_length = args.trace_length,
          batch_size = args.batch_size,
          learning_rate = args.learning_rate)
