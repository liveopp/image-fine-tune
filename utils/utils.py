# -*- coding:utf-8 -*-
"""
一些工具函数
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

input_key = 'input_img'
output_key = 'pred'


def save_model(sess,
               inp_tensor,
               out_tensor,
               output_path):
    '''
    将模型保存成文件
    :param sess:  Tensorflow session
    :param inp_tensor: 网络的输入 tensor
    :param out_tensor: 网络的输出 tensor
    :param output_path: 输出模型文件的路径
    '''

    inputs = {input_key: inp_tensor}
    outputs = {output_key: out_tensor}

    signature_def_map = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.saved_model.signature_def_utils.predict_signature_def(inputs, outputs)
    }

    b = tf.saved_model.builder.SavedModelBuilder(output_path)
    b.add_meta_graph_and_variables(
        sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map=signature_def_map,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))

    b.save()
    return


def write_pb(sess, graph, output_path, pb_fname):
    ''' 将模型保存成文件
    :param sess:  Tensorflow session
    :param graph:  Tensorflow graph
    :param output_path: 输出模型文件的路径
    :param pb_fname: 输出模型文件的文件名
    '''

    origin_def = graph.as_graph_def()
    graph_def = tf.graph_util.convert_variables_to_constants(sess, origin_def, ['output'])
    graph_def = tf.graph_util.remove_training_nodes(graph_def)
    tf.train.write_graph(graph_def, output_path, pb_fname, as_text=False)

    return


def mkdir_p(path):
    """ make a folder in file system """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if os.path.isdir(path):
            pass
        else:
            raise


if __name__ == '__main__':
    pass
