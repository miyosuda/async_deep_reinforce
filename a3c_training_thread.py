# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random

from accum_trainer import AccumTrainer
from game_state import GameState
from game_state import ACTION_SIZE
from game_ac_network import GameACNetwork

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import GRAD_NORM_CLIP

class A3CTrainingThread(object):
  def __init__(self, thread_index, global_network, initial_learning_rate,
               learning_rate_input,
               policy_applier, value_applier,
               max_global_time_step):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    self.local_network = GameACNetwork(ACTION_SIZE)
    self.local_network.prepare_loss(ENTROPY_BETA)

    # policy
    self.policy_trainer = AccumTrainer()
    self.policy_trainer.prepare_minimize( self.local_network.policy_loss,
                                          self.local_network.get_policy_vars(),
                                          GRAD_NORM_CLIP )
    
    self.policy_accum_gradients = self.policy_trainer.accumulate_gradients()
    self.policy_reset_gradients = self.policy_trainer.reset_gradients()
  
    self.policy_apply_gradients = policy_applier.apply_gradients(
        global_network.get_policy_vars(),
        self.policy_trainer.get_accum_grad_list() )

    # value
    self.value_trainer = AccumTrainer()
    self.value_trainer.prepare_minimize( self.local_network.value_loss,
                                         self.local_network.get_value_vars(),
                                         GRAD_NORM_CLIP )
    self.value_accum_gradients = self.value_trainer.accumulate_gradients()
    self.value_reset_gradients = self.value_trainer.reset_gradients()
  

    self.value_apply_gradients = value_applier.apply_gradients(
        global_network.get_value_vars(),
        self.value_trainer.get_accum_grad_list() )
    
    self.sync = self.local_network.sync_from(global_network)
    
    self.game_state = GameState(113 * thread_index)
    
    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0

    # thread0 will record score for TensorBoard
    if self.thread_index == 0:
      self.score_input = tf.placeholder(tf.int32)
      tf.scalar_summary("score", self.score_input)

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values):
    values = []
    sum = 0.0
    for rate in pi_values:
      sum = sum + rate
      value = sum
      values.append(value)
    
    r = random.random() * sum
    for i in range(len(values)):
      if values[i] >= r:
        return i;
    #fail safe
    return len(values)-1

  def _record_score(self, sess, summary_writer, summary_op, score, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      self.score_input: score
    })
    summary_writer.add_summary(summary_str, global_t)
    
  def process(self, sess, global_t, summary_writer, summary_op):
    states = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    # 加算された勾配をリセット
    sess.run( self.policy_reset_gradients )
    sess.run( self.value_reset_gradients )

    # shared から localにweightをコピー
    sess.run( self.sync )

    start_local_t = self.local_t
    
    # 5回ループ
    for i in range(LOCAL_T_MAX):
      pi_ = self.local_network.run_policy(sess, self.game_state.s_t)
      action = self.choose_action(pi_)

      states.append(self.game_state.s_t)
      actions.append(action)
      value_ = self.local_network.run_value(sess, self.game_state.s_t)
      values.append(value_)

      if (self.thread_index == 0) and (self.local_t % 100) == 0:
        print "pi=", pi_
        print " V=", value_

      # gameを実行
      self.game_state.process(action)

      # 実行した結果
      reward = self.game_state.reward
      terminal = self.game_state.terminal

      self.episode_reward += reward

      rewards.append(reward)

      self.local_t += 1

      self.game_state.update()
      
      if terminal:
        terminal_end = True
        print "score=", self.episode_reward

        if self.thread_index == 0:        
          self._record_score(sess, summary_writer, summary_op, self.episode_reward, global_t)
          
        self.episode_reward = 0
        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, self.game_state.s_t)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    # 勾配を算出して加算していく
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      td = R - Vi
      a = np.zeros([ACTION_SIZE])
      a[ai] = 1

      sess.run( self.policy_accum_gradients,
                feed_dict = {
                    self.local_network.s: [si],
                    self.local_network.a: [a],
                    self.local_network.td: [td] } )
      
      sess.run( self.value_accum_gradients,
                feed_dict = {
                    self.local_network.s: [si],
                    self.local_network.r: [R] } )

    cur_learning_rate = self._anneal_learning_rate(global_t)

    sess.run( self.policy_apply_gradients,
              feed_dict = { self.learning_rate_input: cur_learning_rate } )
    sess.run( self.value_apply_gradients,
              feed_dict = { self.learning_rate_input: cur_learning_rate } )

    if (self.thread_index == 0) and (self.local_t % 100) == 0:
      print "TIMESTEP", self.local_t

    # 進んだlocal step数を返す
    diff_local_t = self.local_t - start_local_t
    return diff_local_t
    
