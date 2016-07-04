# -*- coding: utf-8 -*-
import sys
import numpy as np
import gym

import skimage.color
import skimage.transform

from constants import GYM_ENV
from constants import ACTION_SIZE

class GameState(object):
  def __init__(self, display=False, no_op_max=7):
    self._no_op_max = no_op_max
    self._display = display
    
    self.env = gym.make(GYM_ENV)

    print "action space=", self.env.action_space
    
    self.reset()
    
  def _process_frame(self, action, reshape):
    observation, reward, terminal, info = self.env.step(action)
    # observation shape = (210, 160, 3)
    
    grayscale_observation = skimage.color.rgb2gray(observation)
    # shape (210, 160) range = [0.0, 1.0]

    # resize to height=110, width=84
    resized_observation = skimage.transform.resize(grayscale_observation, (110, 84))
    resized_observation = resized_observation.astype(np.float32)
    
    x_t = resized_observation[18:102,:]

    if reshape:
      x_t = np.reshape(x_t, (84, 84, 1))
    return reward, terminal, x_t
    
  def reset(self):
    self.env.reset()
    
    # randomize initial state
    if self._no_op_max > 0:
      no_op = np.random.randint(0, self._no_op_max + 1)
      for _ in range(no_op):
        self.env.step(0)

    _, _, x_t = self._process_frame(0, False)
    
    self.reward = 0
    self.terminal = False
    self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    
  def process(self, action):
    if self._display:
      self.env.render()
    
    r, t, x_t1 = self._process_frame(action, True)

    self.reward = r
    self.terminal = t
    self.s_t1 = np.append(self.s_t[:,:,1:], x_t1, axis = 2)    

  def update(self):
    self.s_t = self.s_t1
