#!/bin/bash

# MPCC Experiment Runner

# System selection
SYS='quadrotor_2D'
TASK='tracking'
ALGO='mpcc'

# Track source: 'builtin' or 'env'
TRACK_SOURCE='env'

# Determine system name
if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

# Select config files
if [ "$TRACK_SOURCE" == 'env' ]; then
    SUFFIX='_env_circle'
else
    SUFFIX=''
fi

# Run experiment
python3 ./mpcc_experiment.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}${SUFFIX}.yaml \
        ./config_overrides/${SYS}/${ALGO}_${SYS}_${TASK}${SUFFIX}.yaml
