import sys
sys.path.append('../')
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
import graphconv.common.tf_util as U


def mlp(input, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        out = tf.expand_dims(out, 1)
        return out


def my_graph_model_policy_network(input1, neighbors, agent_n, num_outputs, scope, reuse=False):
    encoder = mlp
    vec = np.zeros((1, neighbors))
    vec[0][0] = 1
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        x1 = []
        for _ in range(agent_n):
            x1.append(encoder(input1[_], scope))
        x1_ = tf.concat(x1, axis=1)
        out = x1_
        return out


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplemented()


class PlacholderTfInput(TfInput):
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholder."""
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}

class BatchInput(PlacholderTfInput):
    def __init__(self, shape, dtype=tf.float32, name=None):
        """Creates a placeholder for a batch of tensors of a given shape and dtype

        Parameters
        ----------
        shape: [int]
            shape of a single elemenet of the batch
        dtype: tf.dtype
            number representation used for tensor contents
        name: str
            name of the underlying placeholder
        """
        super().__init__(tf.placeholder(dtype, [None] + list(shape), name=name))

num_adversaries = 4
name = "adversaries"
agent_n = 6
neighbor = 2
obs_shape_n = [(34,), (34,), (34,), (34,), (28,), (28,)]
p_input = []
p_input1 = []
p_input2 = []
for i in range(num_adversaries):
    p_input1.append(BatchInput(obs_shape_n[i], name="observation" + str(i)).get())
    p_input2.append(BatchInput([neighbor, num_adversaries], name="adjacency" + str(i)).get())
p2 = mlp(p_input1[0], scope="2_scope", reuse=False)
p2_func_vars = U.scope_vars(U.absolute_scope_name("2_scope"))
p = my_graph_model_policy_network(p_input1[0], 2, 4, 5, scope="p_func")
p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
act = U.function(inputs=[p_input1], outputs=p)

