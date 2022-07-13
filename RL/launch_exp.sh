#!/bin/bash

set -x

python run_trained_agents.py \
--has_cuda=0 \
--safe_max_rho 0.5 0.6 0.7 0.8 0.9 0.95 1.0 1.1 1.2 \
--limit_cs_margin 100. \
--expe_name figure_7_1 \
--nb_process 4 \
--path_agents ADPRL_paper

#--limit_cs_margin 40. 60. 80 100. 120. 140. 160. 180. \
#--expe_name figure_7_2 \
#--safe_max_rho 0.9 \