# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random

from maze_state import MazeState
from maze_ac_network import MazeACNetwork
from a3c_training_thread import A3CTrainingThread
from constants import ACTION_SIZE
from constants import CHECKPOINT_DIR
from constants import PARALLEL_SIZE

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

def choose_best_action(pi_values):
  return np.argmax(pi_values)

def show_value(network, x, y):
  image = np.zeros( (10, 10), dtype=float )
  image[x][y] = 1.0
  s_t = np.reshape(image, (10, 10, 1) )
  
  pi_values = network.run_policy(sess, s_t)
  v_value = network.run_value(sess, s_t)
  
  print "x=",x, " y=",y
  print "pi=",pi_values
  print "v=",v_value
  

def calc_policy(network, x, y):
  image = np.zeros( (10, 10), dtype=float )
  image[x][y] = 1.0
  s_t = np.reshape(image, (10, 10, 1) )
  pi_values = network.run_policy(sess, s_t)
  return pi_values


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
  print "------------"
  for y in range(10):
    show_value(global_network, x, y)

print "======================="

maze_state = MazeState()

step = 0

print "step=", step, ": x=", maze_state.x, " y=", maze_state.y

for _ in range(1000):
  step += 1
  pi = calc_policy(global_network, maze_state.x, maze_state.y)
  #action = choose_best_action(pi)
  action = choose_action(pi)
  maze_state.process(action)
  print "step=", step, ": x=", maze_state.x, " y=", maze_state.y  

  if maze_state.terminal:
    print "reached to goal"
    break

  
  
  


