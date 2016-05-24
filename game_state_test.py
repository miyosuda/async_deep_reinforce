import unittest
import numpy as np

from game_state import GameState

class TestSequenceFunctions(unittest.TestCase):

  def test_process(self):
    game_state = GameState(0)
    
    before_s_t = np.array( game_state.s_t )
    
    for i in range(1000):
      bef1 = game_state.s_t[:,:,1]
      bef2 = game_state.s_t[:,:,2]
      bef3 = game_state.s_t[:,:,3]

      game_state.process(1)
      game_state.update()
      
      aft0 = game_state.s_t[:,:,0]
      aft1 = game_state.s_t[:,:,1]
      aft2 = game_state.s_t[:,:,2]

      # values should be shifted
      self.assertTrue( (bef1.flatten() == aft0.flatten()).all() )
      self.assertTrue( (bef2.flatten() == aft1.flatten()).all() )
      self.assertTrue( (bef3.flatten() == aft2.flatten()).all() )

      # all element should be less [0.0~1.0]
      self.assertTrue( np.less_equal(bef1, 1.0).all() )
      self.assertTrue( np.less_equal(bef2, 1.0).all() )
      self.assertTrue( np.less_equal(bef3, 1.0).all() )
      self.assertTrue( np.greater_equal(bef1, 0.0).all() )
      self.assertTrue( np.greater_equal(bef2, 0.0).all() )
      self.assertTrue( np.greater_equal(bef3, 0.0).all() )

      self.assertTrue( np.less_equal(aft0, 1.0).all() )
      self.assertTrue( np.less_equal(aft1, 1.0).all() )
      self.assertTrue( np.less_equal(aft2, 1.0).all() )
      self.assertTrue( np.greater_equal(aft0, 0.0).all() )
      self.assertTrue( np.greater_equal(aft1, 0.0).all() )
      self.assertTrue( np.greater_equal(aft2, 0.0).all() )

if __name__ == '__main__':
  unittest.main()
