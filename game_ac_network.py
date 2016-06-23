# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Actor-Critic Network (Policy network and Value network)

class GameACNetwork(object):
  def __init__(self,
               action_size,
               device="/cpu:0"):
    self._device = device
    
    with tf.device(self._device):
      self._action_size = action_size
      
      self.W_conv1 = self._conv_weight_variable([8, 8, 4, 16])  # stride=4
      self.b_conv1 = self._conv_bias_variable([16], 8, 8, 4)

      self.W_conv2 = self._conv_weight_variable([4, 4, 16, 32]) # stride=2
      self.b_conv2 = self._conv_bias_variable([32], 4, 4, 16)

      self.W_fc1 = self._fc_weight_variable([2592, 256])
      self.b_fc1 = self._fc_bias_variable([256], 2592)

      # weight for policy output layer
      self.W_fc2 = self._fc_weight_variable([256, action_size])
      self.b_fc2 = self._fc_bias_variable([action_size], 256)

      # weight for value output layer
      self.W_fc3 = self._fc_weight_variable([256, 1])
      self.b_fc3 = self._fc_bias_variable([1], 256)

      # state (input)
      self.s = tf.placeholder("float", [None, 84, 84, 4])
    
      h_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

      h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)

      # policy (output)
      self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
      # value (output)
      v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )

  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])
      
      # policy entropy
      entropy = -tf.reduce_sum(self.pi * tf.log(self.pi), reduction_indices=1)
      
      # policy loss (output)  (add minus, because this is for gradient ascent)
      policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.mul( tf.log(self.pi), self.a ), reduction_indices=1 ) * self.td + entropy * entropy_beta )

      # R (input for value)
      self.r = tf.placeholder("float", [None])
      
      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

      # gradienet of policy and value are summed up
      self.total_loss = policy_loss + value_loss

  def run_policy_and_value(self, sess, s_t):
    pi_out, v_out = sess.run( [self.pi, self.v], feed_dict = {self.s : [s_t]} )
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
    return pi_out[0]

  def run_value(self, sess, s_t):
    v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.op_scope([], name, "GameACNetwork") as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_weight_variable(self, shape):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)

  def _fc_bias_variable(self, shape, input_channels):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)  

  def _conv_weight_variable(self, shape):
    w = shape[0]
    h = shape[1]
    input_channels = shape[2]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)

  def _conv_bias_variable(self, shape, w, h, input_channels):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")
