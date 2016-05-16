# -*- coding: utf-8 -*-

import tensorflow as tf

class AccumTrainer(object):
  def __init__(self,
               device="/cpu:0",
               name="AccumTrainer"):
    self._name = name
    self._device = device

  def _create_accum_grad(self, var):
    """
    Create Variable where to accumulate gradients.
    """
    zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
    name = var.name.replace(":", "_") + "_accum_grad"
    accum_grad = tf.Variable(zero, name=name, trainable=False)
    return accum_grad

  def prepare_minimize(self, loss, var_list):
    with tf.device(self._device):
      var_refs = [v.ref() for v in var_list]
      grads = tf.gradients(
        loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

      self._var_list = var_list
      self._grad_list = grads
      self._accum_grad_list = []
    
      with tf.control_dependencies(None):
        for var in var_list:
          accum_grad = self._create_accum_grad(var)
          self._accum_grad_list.append(accum_grad)

  def get_accum_grad_list(self):
    return self._accum_grad_list

  def accumulate_gradients(self, name=None):
    with tf.device(self._device):
      accumulate_ops = []
    
      with tf.op_scope([], name, self._name) as name:
        for var, grad, accum_grad in zip(self._var_list, self._grad_list, self._accum_grad_list):
          with tf.name_scope("accum_" + var.op.name):
            accumulate_ops.append( tf.assign_add(accum_grad, grad) )
        return tf.group(*accumulate_ops, name=name)

  def reset_gradients(self,  name=None):
    with tf.device(self._device):
      reset_ops = []

      with tf.op_scope([], name, self._name) as name:
        for var, accum_grad in zip(self._var_list, self._accum_grad_list):
          with tf.name_scope("reset_" + var.op.name):
            zero = tf.zeros(accum_grad.get_shape())
            reset = accum_grad.assign(zero)
            reset_ops.append(reset)
        return tf.group(*reset_ops, name=name)
