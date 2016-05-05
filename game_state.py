# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
from ale_python_interface import ALEInterface

from constants import ROM
from constants import ACTION_SIZE

class GameState(object):
  def __init__(self, rand_seed, display=False):
    self.ale = ALEInterface()
    self.ale.setInt('random_seed', rand_seed)

    if display:
      self._setup_display()
    
    self.ale.loadROM(ROM)

    # height=210, width=160
    self.screen = np.empty((210, 160, 1), dtype=np.uint8)
    
    no_action = 0
    
    self.reward = self.ale.act(no_action)
    self.terminal = self.ale.game_over()

    # screenのshapeは、(210, 160, 1)
    self.ale.getScreenGrayscale(self.screen)
    
    # (210, 160)にreshape
    reshaped_screen = np.reshape(self.screen, (210, 160))
    
    # height=110, width=84にリサイズ
    resized_screen = cv2.resize(reshaped_screen, (84, 110))
    
    x_t = resized_screen[18:102,:]
    x_t = x_t.astype(np.float32)
    x_t *= (1.0/255.0)
    self.s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    # 実際に利用するactionのみを集めておく
    self.real_actions = self.ale.getMinimalActionSet()
    
  def _setup_display(self):
    if sys.platform == 'darwin':
      import pygame
      pygame.init()
      self.ale.setBool('sound', False)
    elif sys.platform.startswith('linux'):
      self.ale.setBool('sound', True)
    self.ale.setBool('display_screen', True)
    
  def process(self, action):
    # 18種類のうちの実際に利用するactionに変換
    real_action = self.real_actions[action]
    self.reward = self.ale.act(real_action)
    self.terminal = self.ale.game_over()
    
    # screenのshapeは、(210, 160, 1)
    self.ale.getScreenGrayscale(self.screen)
    
    # (210, 160)にreshape
    reshaped_screen = np.reshape(self.screen, (210, 160))
    
    # height=210, width=160
    
    # height=110, width=84にリサイズ
    resized_screen = cv2.resize(reshaped_screen, (84, 110))
    x_t1 = resized_screen[18:102,:]
    x_t1 = np.reshape(x_t1, (84, 84, 1))
    x_t1 = x_t1.astype(np.float32)    
    x_t1 *= (1.0/255.0)
    
    self.s_t1 = np.append(x_t1, self.s_t[:,:,0:3], axis = 2)
    if self.terminal:
      self.ale.reset_game()

  def update(self):
    self.s_t = self.s_t1
