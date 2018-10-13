# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_optimizer(hypes, learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if yamls.optimizer is not recognized.
    """
    if hypes.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=hypes.adadelta_rho,
            epsilon=hypes.opt_epsilon)
    elif hypes.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=hypes.adagrad_initial_accumulator_value)
    elif hypes.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=hypes.adam_beta1,
            beta2=hypes.adam_beta2,
            epsilon=hypes.opt_epsilon)
    elif hypes.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=hypes.ftrl_learning_rate_power,
            initial_accumulator_value=hypes.ftrl_initial_accumulator_value,
            l1_regularization_strength=hypes.ftrl_l1,
            l2_regularization_strength=hypes.ftrl_l2)
    elif hypes.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=hypes.momentum,
            name='Momentum')
    elif hypes.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=hypes.rmsprop_decay,
            momentum=hypes.momentum,
            epsilon=hypes.opt_epsilon)
    elif hypes.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', hypes.optimizer)
    return optimizer
