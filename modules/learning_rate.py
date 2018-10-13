# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_learning_rate(hypes, num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / hypes.batch_size *
                      hypes.num_epochs_per_decay)
    if hypes.sync_replicas:
        decay_steps /= hypes.replicas_to_aggregate

    if hypes.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(hypes.learning_rate,
                                          global_step,
                                          decay_steps,
                                          hypes.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif hypes.learning_rate_decay_type == 'fixed':
        return tf.constant(hypes.learning_rate, name='fixed_learning_rate')
    elif hypes.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(hypes.learning_rate,
                                         global_step,
                                         decay_steps,
                                         hypes.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         hypes.learning_rate_decay_type)
