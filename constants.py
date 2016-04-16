# -*- coding: utf-8 -*-

GAMMA = 0.95
#GAMMA = 0.99 # discount factor for rewards
LOCAL_T_MAX = 5 # repeat step size
RMSP_EPSILON = 1e-10 # epsilon parameter for RMSProp
#ENTROPY_BETA = 0.001 # entropy regurarlization constant
#ENTROPY_BETA = 0.01 # entropy regurarlization constant
ENTROPY_BETA = 0.1 # entropy regurarlization constant

ROM = "pong.bin"     # action size = 3

ACTION_SIZE = 4 # action size

#PARALLEL_SIZE = 1 # parallel thread size
PARALLEL_SIZE = 8 # parallel thread size
CHECKPOINT_DIR = 'checkpoints'

INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate
INITIAL_ALPHA_LOG_RATE = 1.0 # log_uniform interpolate rate for learning rate
#MAX_TIME_STEP = 4000000 * 8 #
MAX_TIME_STEP = 100000 # 
