# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Actor-Critic Network (Policy network and Value network)

class GameACNetwork(object):
  def __init__(self, action_size):
    with tf.device("/cpu:0"):
      self._action_size = action_size

      self.W_fc1 = self._weight_variable([100, 400])
      self.b_fc1 = self._bias_variable([400])

      # weight for policy output layer
      self.W_fc2 = self._weight_variable([400, action_size])
      self.b_fc2 = self._bias_variable([action_size])

      # weight for value output layer
      self.W_fc3 = self._weight_variable([400, 1])
      self.b_fc3 = self._bias_variable([1])

      # state (input)
      self.s = tf.placeholder("float", [1, 10, 10, 1])
      
      s_flat = tf.reshape(self.s, [1, 100])
      h_fc1 = tf.nn.relu(tf.matmul(s_flat, self.W_fc1) + self.b_fc1)

      # policy (output)
      self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
      # value (output)
      self.v = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3

  def prepare_loss(self, entropy_beta):
    with tf.device("/cpu:0"):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [1, self._action_size])
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [1])
      # policy entropy
      entropy = -tf.reduce_sum(self.pi * tf.log(self.pi))
      # policy loss (output)  (add minus, because this is for gradient ascent)
      # TODO: ここのpolicy_lossのlog(pi)の計算部分が正しいかどうか要検討
      self.policy_loss = -( tf.reduce_sum( tf.mul( tf.log(self.pi), self.a ) ) * self.td + entropy * entropy_beta )

      # R (input for value)
      self.r = tf.placeholder("float", [1])
      # value loss (output)
      self.value_loss = tf.reduce_mean(tf.square(self.r - self.v))

  def run_policy(self, sess, s_t):
    pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
    return pi_out[0]

  def run_value(self, sess, s_t):
    v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
    return v_out[0][0] # output is scalar

  def get_policy_vars(self):
    return [self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2]

  def get_value_vars(self):
    return [self.W_fc1, self.b_fc1,
            self.W_fc3, self.b_fc3]

  def sync_from(self, src_netowrk, name=None):
    src_policy_vars = src_netowrk.get_policy_vars()
    src_value_vars = src_netowrk.get_value_vars()
      
    dst_policy_vars = self.get_policy_vars()
    dst_value_vars = self.get_value_vars()

    sync_ops = []

    with tf.device("/cpu:0"):    
      with tf.op_scope([], name, "GameACNetwork") as name:
        for(src_policy_var, dst_policy_var) in zip(src_policy_vars, dst_policy_vars):
          sync_op = tf.assign(dst_policy_var, src_policy_var)
          sync_ops.append(sync_op)

        for(src_value_var, dst_value_var) in zip(src_value_vars, dst_value_vars):
          sync_op = tf.assign(dst_value_var, src_value_var)
          sync_ops.append(sync_op)
      
        return tf.group(*sync_ops, name=name)

  def _weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

  def _bias_variable(self, shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

  def _save_sub(self, sess, prefix, var, name):
    var_val = var.eval(sess)
    var_val = np.reshape(var_val, (1, np.product(var_val.shape)))        
    np.savetxt('./' + prefix + '_' + name + '.csv', var_val, delimiter=',')

  def save(self, sess, prefix):
    self._save_sub(sess, prefix, self.W_fc1, "W_fc1")
    self._save_sub(sess, prefix, self.b_fc1, "b_fc1")
    self._save_sub(sess, prefix, self.W_fc2, "W_fc2")
    self._save_sub(sess, prefix, self.b_fc2, "b_fc2")
    self._save_sub(sess, prefix, self.W_fc3, "W_fc3")
    self._save_sub(sess, prefix, self.b_fc3, "b_fc3")
