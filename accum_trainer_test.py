# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import accum_trainer

class AccumTrainerTest(tf.test.TestCase):
  def testAccum(self):
    with self.test_session():
      var0 = tf.Variable([1.0, 2.0])
      trainer = accum_trainer.AccumTrainer()
      
      cost = tf.square(var0)
      
      trainer.prepare_minimize(cost, [var0])
      
      accmulate_grad = trainer.accumulate_gradients()
      reset = trainer.reset_gradients()
      
      tf.initialize_all_variables().run()

      # gradの加算を実行
      accmulate_grad.run()
      
      # accmulate_gradしても、var0の中身は変わらない
      self.assertAllClose([1.0, 2.0], var0.eval())
      
      accum_grads = trainer._accum_grad_list
      accum_grad0 = accum_grads[0]

      # gradがaccum_gradへ加算されているのを確認
      self.assertAllClose([2.0, 4.0], accum_grad0.eval())

      # gradの加算を再度実行
      accmulate_grad.run()

      # gradがaccum_gradへさらに加算されているのを確認
      self.assertAllClose([4.0, 8.0], accum_grad0.eval())

      # resetを実行
      reset.run()

      # accum_gradがゼロになっているのを確認
      self.assertAllClose([0.0, 0.0], accum_grad0.eval())
      

if __name__ == "__main__":
  tf.test.main()
