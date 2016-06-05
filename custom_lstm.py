# -*- coding: utf-8 -*-
import tensorflow as tf

class CustomBasicLSTMCell(object):
  """
  Modified original BasicLSTMCell to store matrix and bias.
  """
  def __init__(self, num_units, forget_bias=1.0, input_size=None):
    self._num_units = num_units
    self._input_size = num_units if input_size is None else input_size
    self._forget_bias = forget_bias

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return 2 * self._num_units

  def __call__(self, inputs, state, scope):
    with tf.variable_scope(scope):
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = tf.split(1, 2, state)
      concat = self._linear([inputs, h], 4 * self._num_units)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(1, 4, concat)
      
      # cell
      new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
      # output
      new_h = tf.tanh(new_c) * tf.sigmoid(o)

    return new_h, tf.concat(1, [new_c, new_h])

  def _linear(self, args, output_size, bias_start=0.0):
    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
      total_arg_size += shape[1]
  
    # Now the computation.
    with tf.variable_scope("Linear"):
      # store as a member for copying
      self.matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
      
      res = tf.matmul(tf.concat(1, args), self.matrix)
      # store as a member for copying
      self.bias = tf.get_variable( "Bias",
                                   [output_size],
                                   initializer=tf.constant_initializer(bias_start))
    return res + self.bias


"""
class CustomLSTMCell(RNNCell):
  def __init__(self,
               num_units,
               input_size=None,
               use_peepholes=False,
               cell_clip=None,
               initializer=None,
               num_proj=None,
               num_unit_shards=1,
               num_proj_shards=1,
               forget_bias=1.0):
    
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated." % self)
 
    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias

    if num_proj:
      self._state_size = num_units + num_proj
      self._output_size = num_proj
    else:
      self._state_size = 2 * num_units
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope=None):
    num_proj = self._num_units if self._num_proj is None else self._num_proj

    c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
    m_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    
    with vs.variable_scope(scope or type(self).__name__,
                           initializer=self._initializer):  # "LSTMCell"
      
      concat_w = _get_concat_variable(
          "W", [input_size.value + num_proj, 4 * self._num_units],
          dtype, self._num_unit_shards)

      b = vs.get_variable(
          "B", shape=[4 * self._num_units],
          initializer=tf.zeros_initializer, dtype=dtype)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      cell_inputs = tf.concat(1, [inputs, m_prev])
      lstm_matrix = nn_ops.bias_add(math_ops.matmul(cell_inputs, concat_w), b)
      i, j, f, o = tf.split(1, 4, lstm_matrix)

      # Diagonal connections
      if self._use_peepholes:
        w_f_diag = vs.get_variable(
            "W_F_diag", shape=[self._num_units], dtype=dtype)
        w_i_diag = vs.get_variable(
            "W_I_diag", shape=[self._num_units], dtype=dtype)
        w_o_diag = vs.get_variable(
            "W_O_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
             sigmoid(i + w_i_diag * c_prev) * tanh(j))
      else:
        c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * tanh(j))

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type

      if self._use_peepholes:
        m = sigmoid(o + w_o_diag * c) * tanh(c)
      else:
        m = sigmoid(o) * tanh(c)

      if self._num_proj is not None:
        concat_w_proj = _get_concat_variable(
            "W_P", [self._num_units, self._num_proj],
            dtype, self._num_proj_shards)

        m = math_ops.matmul(m, concat_w_proj)

    return m, tf.concat(1, [c, m])

def _get_concat_variable(name, shape, dtype, num_shards):
  # Get a sharded variable concatenated into one tensor.
  sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
  if len(sharded_variable) == 1:
    return sharded_variable[0]

  concat_name = name + "/concat"
  concat_full_name = vs.get_variable_scope().name + "/" + concat_name + ":0"
  for value in ops.get_collection(ops.GraphKeys.CONCATENATED_VARIABLES):
    if value.name == concat_full_name:
      return value

  concat_variable = array_ops.concat(0, sharded_variable, name=concat_name)
  ops.add_to_collection(ops.GraphKeys.CONCATENATED_VARIABLES,
                        concat_variable)
  return concat_variable

def _get_sharded_variable(name, shape, dtype, num_shards):
  # Get a list of sharded variables with the given dtype.
  if num_shards > shape[0]:
    raise ValueError("Too many shards: shape=%s, num_shards=%d" %
                     (shape, num_shards))
  unit_shard_size = int(math.floor(shape[0] / num_shards))
  remaining_rows = shape[0] - unit_shard_size * num_shards

  shards = []
  for i in range(num_shards):
    current_size = unit_shard_size
    if i < remaining_rows:
      current_size += 1
    shards.append(vs.get_variable(name + "_%d" % i, [current_size] + shape[1:],
                                  dtype=dtype))
  return shards
"""
