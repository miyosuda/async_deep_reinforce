# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import random

from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import USE_LSTM

# use CPU for weight visualize tool
device = "/cpu:0"

if USE_LSTM:
  global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)
else:
  global_network = GameACFFNetwork(ACTION_SIZE, -1, device)

training_threads = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
else:
  print("Could not find old checkpoint")
  
W_conv1 = sess.run(global_network.W_conv1)

# show graph of W_conv1
fig, axes = plt.subplots(4, 16, figsize=(12, 6),
             subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for ax,i in zip(axes.flat, range(4*16)):
  inch = i//16
  outch = i%16
  img = W_conv1[:,:,inch,outch]
  ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
  ax.set_title(str(inch) + "," + str(outch))

plt.show()

