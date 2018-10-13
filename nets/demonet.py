# -*- coding:utf-8 -*-
"""
Inception v3的变种网络

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def ss_dishnet(inputs,
               num_classes=1001,
               is_training=True,
               scope='ss_dishnet'):
    """
    输入：
    - inputs (Tensor): 4D Tensor, 形状为：[batch_size, height, width, num_channels]
    - num_classes (int): 类别数量
    - is_training (bool): 是否为训练模式
    - scope (str): tensorflow的变量scope

    输出：
    - logits (Tensor): 每类的原始分数(logits)，需要经过softmax得到最后的分数

    备忘：

    Padding默认设置为有，目的是保留张量的形状.
    """
    conv2d_cnt = 0
    maxpool_cnt = 0
    avgpool_cnt = 0

    def conv2d_bn(x,
                  filters,
                  num_row,
                  num_col,
                  padding='same',
                  strides=(1, 1),
                  name=None):
        nonlocal conv2d_cnt
        conv2d_cnt += 1
        if name is not None:
            bn_name = 'batch_normalization_{}'.format(name)
            conv_name = 'conv2d_{}'.format(name)
            relu_name = 'activation_{}'.format(name)
        else:
            bn_name = 'batch_normalization_{}'.format(conv2d_cnt)
            conv_name = 'conv2d_{}'.format(conv2d_cnt)
            relu_name = 'activation_{}'.format(conv2d_cnt)

        x = tf.layers.conv2d(
            x,
            filters,
            (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)

        x = tf.layers.batch_normalization(x, training=is_training,
                                          scale=False, name=bn_name)

        x = tf.nn.relu(x, name=relu_name)
        return x

    def max_pool2d(inputs,
                pool_size,
                strides,
                padding='valid'):
        nonlocal maxpool_cnt
        maxpool_cnt += 1
        x = tf.layers.max_pooling2d(inputs, pool_size, strides, padding,
                                    name='max_pooling2d_{}'.format(maxpool_cnt))
        return x

    def avg_pool2d(inputs,
                   pool_size,
                   strides,
                   padding='valid'):
        nonlocal avgpool_cnt
        avgpool_cnt += 1
        x = tf.layers.average_pooling2d(inputs, pool_size, strides, padding,
                                        name='average_pooling2d_{}'.format(avgpool_cnt))
        return x

    channel_axis = 3
    with tf.variable_scope(scope) as sc:
        x = conv2d_bn(inputs, 32, 3, 3, strides=(2, 2), padding='valid')
        x = conv2d_bn(x, 32, 3, 3, padding='valid')
        x = conv2d_bn(x, 64, 3, 3)
        x = max_pool2d(x, (3, 3), strides=(2, 2))

        x = conv2d_bn(x, 80, 1, 1, padding='valid')
        x = conv2d_bn(x, 192, 3, 3, padding='valid')
        x = max_pool2d(x, (3, 3), strides=(2, 2))

        # 分叉阶段0
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = avg_pool2d(x, (3, 3), strides=(1, 1), padding='same')
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = tf.concat(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed0')

        # 分叉阶段1
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = avg_pool2d(x, (3, 3), strides=(1, 1), padding='same')
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = tf.concat(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed1')

        # 分叉阶段2
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = avg_pool2d(x, (3, 3), strides=(1, 1), padding='same')
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = tf.concat(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed2')

        # 分叉阶段3
        branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = max_pool2d(x, (3, 3), strides=(2, 2))
        x = tf.concat(
            [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

        # 分叉阶段4
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 128, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = avg_pool2d(x, (3, 3), strides=(1, 1), padding='same')
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = tf.concat(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed4')

        # 分叉阶段5，6
        for i in range(2):
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch7x7 = conv2d_bn(x, 160, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = avg_pool2d(x, (3, 3), strides=(1, 1), padding='same')
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = tf.concat(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(5 + i))

        # 分叉阶段7
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 192, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = avg_pool2d(x, (3, 3), strides=(1, 1), padding='same')
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = tf.concat(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed7')

        # 分叉阶段8
        branch3x3 = conv2d_bn(x, 192, 1, 1)
        branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

        branch7x7x3 = conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = max_pool2d(x, (3, 3), strides=(2, 2))
        x = tf.concat(
            [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

        # 分叉阶段9，10
        for i in range(2):
            branch1x1 = conv2d_bn(x, 320, 1, 1)

            branch3x3 = conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = tf.concat(
                [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

            branch3x3dbl = conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = tf.concat(
                [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

            branch_pool = avg_pool2d(x, (3, 3), strides=(1, 1), padding='same')
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = tf.concat(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(9 + i))

        x = tf.layers.separable_conv2d(x, 2048, [3, 3], padding='same', use_bias=False,
                                       name='post_conv0')
        x = tf.layers.batch_normalization(x, training=is_training, name='post_batchnorm0')
        x = tf.nn.relu(x)
        x = tf.reduce_mean(x, axis=[1, 2], name='ss-top1/Mean', keepdims=True)

        x = tf.layers.conv2d(x, num_classes, [1, 1], name='logits')
        logits = tf.squeeze(x, [1, 2], name='SpatialSqueeze')
        return logits


def load_graph(graph_pb_path, input_tensor):
    with open(graph_pb_path, 'rb') as f:
        content = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(content)
    bottleneck_tensor = tf.import_graph_def(graph_def,
                                            input_map={'input_1:0': input_tensor},
                                            return_elements=['ss-top1/Mean:0'],
                                            name='')
    return bottleneck_tensor[0]


def food_net_from_pb(image, num_classes, pb_file):
    bottleneck_tensor = load_graph(pb_file, image)
    bottleneck_tensor = tf.reshape(bottleneck_tensor, [-1, 1, 1, 2048])
    logits = tf.layers.conv2d(bottleneck_tensor, num_classes, [1, 1], name='logits')
    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
    return logits
