# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os

from game_state import ACTION_SIZE
from game_ac_network import GameACNetwork
from a3c_training_thread import A3CTrainingThread

PARALLEL_SIZE = 8
INITIAL_ALPHA_LOW = 1e-4
INITIAL_ALPHA_HIGH = 1e-2
INITIAL_ALPHA_LOG_RATE = 0.5 # 1e-3
#INITIAL_ALPHA_LOG_RATE = 0.25 # 1e-4
MAX_TIME_STEP = 4000000 * 8
CHECKPOINT_DIR = 'checkpoints'


def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

global_t = 0

stop_requested = False

global_network = GameACNetwork(ACTION_SIZE)

training_threads = []

for i in range(PARALLEL_SIZE):
  training_thread = A3CTrainingThread(i, global_network, initial_learning_rate, MAX_TIME_STEP)
  training_threads.append(training_thread)

# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

init = tf.initialize_all_variables()
sess.run(init)

# init or load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print "checkpoint loaded:", checkpoint.model_checkpoint_path
else:
  print "Could not find old checkpoint"

def train_function(parallel_index):
  global global_t
  
  training_thread = training_threads[parallel_index]
  
  while True:
    if stop_requested:
      break
    if global_t > MAX_TIME_STEP:
      break

    diff_global_t = training_thread.process(sess, global_t)
    global_t += diff_global_t

def signal_handler(signal, frame):
  global stop_requested
  print('You pressed Ctrl+C!')
  stop_requested = True
  
train_threads = []
for i in range(PARALLEL_SIZE):
  train_threads.append(threading.Thread(target=train_function, args=(i,)))
  
signal.signal(signal.SIGINT, signal_handler)

for t in train_threads:
  t.start()

print('Press Ctrl+C to stop')
signal.pause()

print('Now saving data. Please wait')
  
for t in train_threads:
  t.join()

if not os.path.exists(CHECKPOINT_DIR):
  os.mkdir(CHECKPOINT_DIR)  

saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)

