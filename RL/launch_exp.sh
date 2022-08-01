#!/bin/bash

set -x

python run_trained_agents.py \
--has_cuda=0 \
--safe_max_rho 0.2 \
--limit_cs_margin 100. \
--expe_name aeMLP \
--nb_process 4 \
--path_agents ADPRL_paper \
--training_iter 1000000 2000000 3000000 4000000 5000000 6000000 \
--agent_name AEMlpPolicy_20220728_161820 \

#--limit_cs_margin 40. 60. 80 100. 120. 140. 160. 180. \
#--expe_name figure_7_2 \
#--safe_max_rho 1.0 \