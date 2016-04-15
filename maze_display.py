# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random

from maze_state import MazeState
ACTION_SIZE = 4
from game_ac_network import GameACNetwork
from a3c_training_thread import A3CTrainingThread

PARALLEL_SIZE = 1
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

def calc_value(network, x, y):
  image = np.zeros( (10, 10), dtype=float )
  image[x][y] = 1.0
  s_t = np.reshape(image, (10, 10, 1) )
  
  pi_values = network.run_policy(sess, s_t)
  v_value = network.run_value(sess, s_t)
  
  print "x=", x, " y=", y
  print "pi=", pi_values
  print "v=", v_value


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


for x in range(10):
  for y in range(10):
    calc_value(global_network, x, y)

"""
game_state = MazeState()

for _ in range(100):
  pi_values = global_network.run_policy(sess, game_state.s_t)
  v_value = global_network.run_value(sess, game_state.s_t)
  
  print "pi_values=", pi_values
  print "v_value=", v_value

  #print "state=", game_state.s_t

  action = choose_action(pi_values)
  print "action=", action
  
  game_state.process(action)

  game_state.update()
"""
