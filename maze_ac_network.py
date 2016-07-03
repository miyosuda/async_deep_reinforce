# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# Actor-Critic Network (Policy network and Value network)

class MazeACNetwork(object):
  def __init__(self,
               action_size,
               device="/cpu:0"):
    self._device = device
    self._action_size = action_size
    
    with tf.device(device):
      self.W_fc1 = self._weight_variable([100, 800])
      self.b_fc1 = self._bias_variable([800])

      # weight for policy output layer
      self.W_fc2 = self._weight_variable([800, action_size])
      self.b_fc2 = self._bias_variable([action_size])

      # weight for value output layer
      self.W_fc3 = self._weight_variable([800, 1])
      self.b_fc3 = self._bias_variable([1])

      # state (input)
      self.s = tf.placeholder("float", [None, 10, 10, 1])
      
      s_flat = tf.reshape(self.s, [-1, 100])
      h_fc1 = tf.nn.relu(tf.matmul(s_flat, self.W_fc1) + self.b_fc1)

      # policy (output)
      self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
      # value (output)
      self.v = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3

  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])
      entropy = -tf.reduce_sum(self.pi * tf.log(self.pi), reduction_indices=1)
      
      # policy loss (output)  (add minus, because this is for gradient ascent)
      self.policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.mul( tf.log(self.pi), self.a ), reduction_indices=1 ) * self.td + entropy * entropy_beta )

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
    return [self.W_fc1, self.b_fc1,
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

  def _weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

  def _bias_variable(self, shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")
