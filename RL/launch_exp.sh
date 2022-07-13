#!/bin/bash

set -x
python run_trained_agents.py \
--has_cuda 0 \
--expe_name $1 \
--path_agents ADPRL_paper \
--safe_max_rho 0.95 \
--limit_cs_margin 100 \
--training_iter 1000000 2000000 3000000 4000000 5000000 6000000 7000000 8000000 9000000 10000000
--agent_name PPO_agent0_20220709_152030