# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import random

from game_state import GameState
from game_ac_network import GameACNetwork
from a3c_training_thread import A3CTrainingThread
from constants import ACTION_SIZE

PARALLEL_SIZE = 8
CHECKPOINT_DIR = 'checkpoints'

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
  
W_conv1 = sess.run(global_network.W_conv1)

# show graph of W_conv1
fig, axes = plt.subplots(4, 16, figsize=(12, 6),
             subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for ax,i in zip(axes.flat, range(4*16)):
  inch = i/16
  outch = i%16
  img = W_conv1[:,:,inch,outch]
  ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
  ax.set_title(str(inch) + "," + str(outch))

plt.show()

