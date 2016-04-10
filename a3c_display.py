# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random

from game_state import GameState
from game_state import ACTION_SIZE
from game_ac_network import GameACNetwork
from a3c_training_thread import A3CTrainingThread

PARALLEL_SIZE = 8
CHECKPOINT_DIR = 'checkpoints'

def choose_action(pi_values):
  values = []
  sum = 0.0
  for rate in pi_values:
    sum = sum + rate
    value = sum
    values.append(value)
    
  r = random.random() * sum
  for i in range(len(values)):
    if values[i] >= r:
      return i;
  #fail safe
  return len(values)-1

global_network = GameACNetwork(ACTION_SIZE)

training_threads = []
for i in range(PARALLEL_SIZE):
  training_thread = A3CTrainingThread(i, global_network, 1.0, 8000000)
  training_threads.append(training_thread)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print "checkpoint loaded:", checkpoint.model_checkpoint_path
else:
  print "Could not find old checkpoint"

game_state = GameState(0, display=True)

while True:
  pi_values = global_network.run_policy(sess, game_state.s_t)

  action = choose_action(pi_values)
  game_state.process(action)

  game_state.update()

