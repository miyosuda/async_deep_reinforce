# -*- coding: utf-8 -*-
import sys
import numpy as np

"""
Simple 10x10 2D maze.

When the agent takes shortest path, the total reward will be 17.
(-1 * 18 + 1)
-1 for each step and +1 for goal.
"""


"""
    + + + + + + + W + G
    + + W + + + + W + +
    S + W + + + + W + +
    + + W + W W + W + +
    + + W + + W + + + +
    + + W + + W + + + +
    + + + + + W W W + +
    + + + + + + + + W W
    + + W + + + + + + +
    + + W + + + + + + +
"""

WALL = 0.5
PLAYER = 1.0

class MazeState(object):
  def __init__(self):
    self._maze_image = self._create_maze()    
    self._reset()

  def _reset(self):
    self.x = 0
    self.y = 2
    self.reward = 0
    self.terminal = True
    self.s_t = np.reshape( self._move(0, 0), (10, 10, 1) )

  def _create_maze(self):
    image = np.zeros( (10, 10), dtype=float )

    image[2][1] = WALL
    image[2][2] = WALL
    image[2][3] = WALL
    image[2][4] = WALL
    image[2][5] = WALL
    
    image[2][8] = WALL
    image[2][9] = WALL

    image[4][3] = WALL

    image[5][3] = WALL
    image[5][4] = WALL
    image[5][5] = WALL
    image[5][6] = WALL

    image[6][6] = WALL

    image[7][0] = WALL
    image[7][1] = WALL
    image[7][2] = WALL
    image[7][3] = WALL
    image[7][6] = WALL

    image[8][6] = WALL

    image[9][6] = WALL
    return image

  def _move(self, dx, dy):
    new_x = self.x + dx
    new_y = self.y + dy

    if new_x < 0:
      new_x = 0
    if new_y < 0:
      new_y = 0
    if new_x > 9:
      new_x = 9
    if new_y > 9:
      new_y = 9

    is_wall = (self._maze_image[new_x][new_y] == WALL)
    if is_wall:
      new_x = self.x
      new_y = self.y
    self.x = new_x
    self.y = new_y

    #print "x=", self.x, " y=", self.y
    image = np.zeros( (10, 10), dtype=float )
    image[self.x][self.y] = PLAYER
    return image

  def process(self, action):
    dx = 0
    dy = 0
    if action == 0: # UP
      dy = -1
    if action == 1: # DOWN
      dy = 1
    if action == 2: # LEFT
      dx = -1
    if action == 3: # RIGHT
      dx = 1
    # otherwize don't move

    image = self._move(dx, dy)

    self.s_t1 = np.reshape(image, (10, 10, 1) )
    self.terminal = (self.x == 9 and self.y == 0)

    if self.terminal:
      self.reward = 1
    else:
      self.reward = -1

  def reset(self):
    self._reset()

  def update(self):
    if self.terminal:
      self._reset()
    else:
      self.s_t = self.s_t1
