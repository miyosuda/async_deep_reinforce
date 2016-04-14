import unittest
import numpy as np

from maze_state import MazeState

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class TestMazeState(unittest.TestCase):

  def test_process(self):
    maze_state = MazeState()

    self.assertTrue( maze_state.s_t.shape == (10,10,1) )
    self.assertTrue( maze_state.s_t[0][2][0] == 1.0 ) # player
    self.assertTrue( maze_state.s_t[2][1][0] == 0.5 ) # wall
    
    maze_state.process(UP)

    self.assertTrue( maze_state.s_t1.shape == (10,10,1) )
    self.assertTrue( maze_state.s_t1[0][2][0] == 0.0 ) # old player pos
    self.assertTrue( maze_state.s_t1[0][1][0] == 1.0 ) # new player pos

    self.assertTrue( maze_state.reward == -1 )
    self.assertTrue( maze_state.terminal == False )

    maze_state.update()

    self.assertTrue( maze_state.s_t.shape == (10,10,1) )
    self.assertTrue( maze_state.s_t[0][2][0] == 0.0 ) # old player pos
    self.assertTrue( maze_state.s_t[0][1][0] == 1.0 ) # new player pos

    maze_state.process(UP)
    maze_state.update()

    self.assertTrue( maze_state.x == 0 )
    self.assertTrue( maze_state.y == 0 )

    for _ in range(6):
      maze_state.process(RIGHT)
      maze_state.update()

    self.assertTrue( maze_state.x == 6 )
    self.assertTrue( maze_state.y == 0 )

    for _ in range(4):
      maze_state.process(DOWN)
      maze_state.update()

    self.assertTrue( maze_state.x == 6 )
    self.assertTrue( maze_state.y == 4 )

    for _ in range(3):
      maze_state.process(RIGHT)
      maze_state.update()

    for _ in range(3):
      maze_state.process(UP)
      maze_state.update()

    maze_state.process(UP)

    self.assertTrue( maze_state.s_t1[9][0][0] == 1.0 ) # player pos
    self.assertTrue( maze_state.terminal == True )
    self.assertTrue( maze_state.reward == 1 )


if __name__ == '__main__':
  unittest.main()
