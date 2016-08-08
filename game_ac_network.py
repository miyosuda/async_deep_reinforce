# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from custom_lstm import CustomBasicLSTMCell

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
  def __init__(self,
               action_size,
               device="/cpu:0"):
    self._device = device
    self._action_size = action_size

  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])

      # avoid NaN with clipping when value in pi becomes zero
      log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))
      
      # policy entropy
      entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)
      
      # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
      policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.mul( log_pi, self.a ), reduction_indices=1 ) * self.td + entropy * entropy_beta )

      # R (input for value)
      self.r = tf.placeholder("float", [None])
      
      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

      # gradienet of policy and value are summed up
      self.total_loss = policy_loss + value_loss

  def run_policy_and_value(self, sess, s_t):
    raise NotImplementedError()
    
  def run_policy(self, sess, s_t):
    raise NotImplementedError()

  def run_value(self, sess, s_t):
    raise NotImplementedError()    

  def get_vars(self):
    raise NotImplementedError()

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

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               device="/cpu:0"):
    GameACNetwork.__init__(self, action_size, device)
    
    with tf.device(self._device):
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

# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    GameACNetwork.__init__(self, action_size, device)    

    with tf.device(self._device):
      self.W_conv1 = self._conv_weight_variable([8, 8, 4, 16])  # stride=4
      self.b_conv1 = self._conv_bias_variable([16], 8, 8, 4)

      self.W_conv2 = self._conv_weight_variable([4, 4, 16, 32]) # stride=2
      self.b_conv2 = self._conv_bias_variable([32], 4, 4, 16)

      self.W_fc1 = self._fc_weight_variable([2592, 256])
      self.b_fc1 = self._fc_bias_variable([256], 2592)

      # lstm
      self.lstm = CustomBasicLSTMCell(256)

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
      # h_fc1 shape=(5,256)

      h_fc1_reshaped = tf.reshape(h_fc1, [1,-1,256])
      # h_fc_reshaped = (1,5,256)

      # place holder for LSTM unrolling time step size.
      self.step_size = tf.placeholder(tf.float32, [1])

      self.initial_lstm_state = tf.placeholder(tf.float32, [1, self.lstm.state_size])
      
      scope = "net_" + str(thread_index)

      # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
      # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
      # Unrolling step size is applied via self.step_size placeholder.
      # When forward propagating, step_size is 1.
      # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
      lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                        h_fc1_reshaped,
                                                        initial_state = self.initial_lstm_state,
                                                        sequence_length = self.step_size,
                                                        time_major = False,
                                                        scope = scope)

      # lstm_outputs: (1,5,256) for back prop, (1,1,256) for forward prop.
      
      lstm_outputs = tf.reshape(lstm_outputs, [-1,256])

      # policy (output)
      self.pi = tf.nn.softmax(tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2)
      
      # value (output)
      v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )

      self.reset_state()
      
  def reset_state(self):
    self.lstm_state_out = np.zeros([1, self.lstm.state_size])

  def run_policy_and_value(self, sess, s_t):
    # This run_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    pi_out, v_out, self.lstm_state_out = sess.run( [self.pi, self.v, self.lstm_state],
                                                   feed_dict = {self.s : [s_t],
                                                                self.initial_lstm_state : self.lstm_state_out,
                                                                self.step_size : [1]} )
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    # This run_policy() is used for displaying the result with display tool.    
    pi_out, self.lstm_state_out = sess.run( [self.pi, self.lstm_state],
                                            feed_dict = {self.s : [s_t],
                                                         self.initial_lstm_state : self.lstm_state_out,
                                                         self.step_size : [1]} )
                                            
    return pi_out[0]

  def run_value(self, sess, s_t):
    # This run_value() is used for calculating V for bootstrapping at the 
    # end of LOCAL_T_MAX time step sequence.
    # When next sequcen starts, V will be calculated again with the same state using updated network weights,
    # so we don't update LSTM state here.
    prev_lstm_state_out = self.lstm_state_out
    v_out, _ = sess.run( [self.v, self.lstm_state],
                         feed_dict = {self.s : [s_t],
                                      self.initial_lstm_state : self.lstm_state_out,
                                      self.step_size : [1]} )
    
    # roll back lstm state
    self.lstm_state_out = prev_lstm_state_out
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.lstm.matrix, self.lstm.bias,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]
