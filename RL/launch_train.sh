#!/bin/bash

set -x

python agent_experiments.py \
--has_cuda 0 \
--chronics_name="2050-02-14" \
--agent_name Chron_20500214_$1 \
--training_iter 10000000