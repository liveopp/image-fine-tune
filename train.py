# -*- coding:utf-8 -*-
'''
训练工具入口
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from datasets.dataset import ImageData
from nets import dishnet1_7 as dishnet
from utils.utils import write_pb, save_model, mkdir_p


def build_net(images, labels, params):
    """ 创建一个网络的实例

    :param images:
    :param labels:
    :param params: a python dict with params for building a net
    :return: 一个神经网络的实例
    """

    # recover param from the input dict
    num_classes = params.get('num_classes', 1001)
    checkpoint_path = params.get('checkpoint_path', None)
    pretrain_model = params.get('pretrain_model', None)
    exclude = params.get('exclude', ['InceptionV3/Logits', 'InceptionV3/AuxLogits'])
    adam_beta1 = params.get('adam_beta1', 0.99)
    is_full_train = params.get('is_full_train', False)

    assert os.path.exists(checkpoint_path) or os.path.exists(pretrain_model), \
        'Please make sure one of them exists: checkpoint_path: {}, pretrained_model: {}'.format(checkpoint_path, pretrain_model)
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

    logits = dishnet.ss_dishnet(images, num_classes, is_training, scope='')

    net = {
        'learning_rate': learning_rate,
        'is_training': is_training,
        'predictions': tf.argmax(input=logits, axis=1, output_type=labels.dtype),
        'probabilities': tf.nn.softmax(logits, name='output'),
        'loss': tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits),
    }

    # 计算基于交叉熵的损失函数
    loss = tf.losses.get_total_loss()
    eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels,
                                                       predictions=net['predictions'],
                                                       name='acc_op'),
                       'accuracy_top_5': tf.metrics.mean(tf.nn.in_top_k(
                           predictions=net['probabilities'], targets=labels, k=5),
                           name='acc_top5_op'),
                      }

    # 定义 TOP1和TOP5准确率
    net['accuracy'] = tf.reduce_mean(tf.to_float(tf.equal(net['predictions'], labels)))
    net['accuracy_top_5'] = tf.reduce_mean(tf.to_float(
        tf.nn.in_top_k(predictions=logits, targets=labels, k=5)))

    for metric in ['loss', 'accuracy', 'accuracy_top_5']:
        tf.summary.scalar(metric, net[metric])
    net['eval_metric'] = eval_metric_ops

    # restore from weights trained by ImageNet
    if pretrain_model and os.path.exists(pretrain_model) and 'ckpt' in pretrain_model \
            and tf.train.latest_checkpoint(checkpoint_path) is None:
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=exclude)
        tf.train.init_from_checkpoint(pretrain_model,
                                      {v.name.split(':')[0]: v for v in variables_to_restore})
    else:
        raise FileNotFoundError('pretrained weights not found in {}'.format(pretrain_model))

    global_step = tf.train.get_or_create_global_step()
    net['step'] = global_step
    if is_full_train:
        last_layer = None
    else:
        last_layer = []
        for s in exclude:
            last_layer += tf.contrib.framework.get_variables(s)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=adam_beta1)

    # add update op for batch normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss, global_step=global_step,
                                      var_list=last_layer)
    net['train_op'] = train_op
    net['summary'] = tf.summary.merge_all()

    return net


def train_and_evaluate(sess, train_iterator, val_iterator,
                       train_writer, eval_writer, net,
                       saver, hypes, lr):
    """ the function runs training and just-on-time evaluation
    :param sess: a Tensorflow session instance
    :param train_iterator: training dataset iterator
    :param val_iterator: eval dataset iterator
    :param train_writer: writer to log the training logs
    :param eval_writer: writer to log the evaluation logs
    :param net: the nnet definition
    :param saver: a handle to save/restore the model
    :param hypes: a python dict with all configuration and hype params
    :param lr: learning rate
    """

    # recover variables from the input dict (config file)
    save_ckpt_step = hypes["train"]["save_ckpt_step"]
    model_name = hypes["train"]["model_name"]
    stable_steps = hypes["train"]["stable_steps"]
    lr_decay_rate = hypes["train"]["lr_decay_rate"]
    minimum_lr = hypes["train"]["minimum_lr"]
    log_step = hypes["train"]["log_step"]
    eval_step = hypes["train"]["eval_step"]
    eval_max_batches = hypes["train"]["eval_max_batches"]
    ckpt_dir = hypes["train"]["ckpt_dir"]

    sess.run(train_iterator.initializer)
    sess.run(val_iterator.initializer)
    train_tensors = {k: net[k] for k in ('loss', 'accuracy', 'accuracy_top_5',
                                         'summary', 'train_op', 'step') if k in net}
    val_tensors = {k: net[k] for k in ('loss', 'accuracy', 'accuracy_top_5',
                                       'summary') if k in net}

    is_training, handle = net['is_training'], net['handle']
    train_handle, val_handle = net['train_handle'], net['val_handle']
    learning_rate = net['learning_rate']
    lr_exp = tf.train.exponential_decay(lr, net['step'],
                                        stable_steps, lr_decay_rate,
                                        staircase=True)

    while True:
        try:
            lr = max(sess.run(lr_exp), minimum_lr)
            train_result = sess.run(train_tensors, feed_dict={is_training: True,
                                                              handle: train_handle,
                                                              learning_rate: lr})
            train_writer.add_summary(train_result['summary'], train_result['step'])

            step = train_result['step']

            # save log per X steps
            if step % log_step == 0:
                tf.logging.info('step {} -- train loss: {:.3f} acc_top1: {:.3f} acc_top5: {:.3f}'.format(
                    step, train_result['loss'], train_result['accuracy'], train_result['accuracy_top_5']))

            if step % eval_step == 0:
                eval_result = fully_evaluate(sess, val_iterator, net['eval_metric'],
                                             {is_training: False, handle: val_handle},
                                             eval_max_batches)
                tf.logging.info('step {} -- on validation set: acc_top1: {:.3f} acc_top5: {:.3f}'.format(
                    step, eval_result['accuracy'], eval_result['accuracy_top_5']))
                eval_result = sess.run(val_tensors, feed_dict={is_training: False,
                                                               handle: val_handle})
                eval_writer.add_summary(eval_result['summary'], step)

            # save checkpoint per X steps
            if step % save_ckpt_step == 0:
                saver.save(sess, os.path.join(ckpt_dir, model_name),
                           global_step=step, write_meta_graph=False)

        except tf.errors.OutOfRangeError:
            # Start all over again.
            sess.run(train_iterator.initializer)
        except KeyboardInterrupt:
            # 在中断训练前，保存checkpoint
            saver.save(sess, os.path.join(ckpt_dir, model_name),
                       global_step=step, write_meta_graph=False)
            break


def fully_evaluate(sess, val_iterator, eval_metric, feed_dict, max_batches):
    """ 跑一次完整的evaluation
    :param sess: a Tensorflow session instance
    :param val_iterator: evaluation dataset iterator
    :param eval_metric:  evaluation metric, e.g. top1/top5 accuracy
    :param feed_dict: the feed dict in Tensorflow
    :param max_batches:
    """
    sess.run(tf.local_variables_initializer())
    eval_op = [eval_metric[k][1] for k in eval_metric]
    eval_value = dict([(k, eval_metric[k][0]) for k in eval_metric])
    count = 0
    while True:
        try:
            sess.run(eval_op, feed_dict=feed_dict)
            count += 1
            if count > max_batches > 0:
                break
        except tf.errors.OutOfRangeError:
            if max_batches > 0:
                sess.run(val_iterator.initializer)
            else:
                break

    return sess.run(eval_value)


def export_pb_file(params, output_path, pb_fname):
    """ 将模型frozen之后到处成PB文件

    :param params: 训练所用参数
    :param output_path: 输出的目录
    :param pb_fname: 输出的PB文件名
    """
    num_classes = params['num_classes']
    ckpt_dir = params['checkpoint_path']
    graph = tf.Graph()
    with graph.as_default():
        images = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='input')
        logits = dishnet.ss_dishnet(images, num_classes, False, scope='')
        out_tensor = tf.nn.softmax(logits, name='output')
        saved_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        assert saved_ckpt is not None
        with tf.Session(graph=graph) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, saved_ckpt)
            if pb_fname.endswith('.pb'):
                write_pb(sess, graph, output_path, pb_fname)
            else:
                save_model(sess, images, out_tensor, os.path.join(output_path, pb_fname))


def main(hypes):
    """ 运行训练的入口函数
    :param hypes: a python dict with all configuration and hype params
    """

    # 设置log级别
    tf.logging.set_verbosity(tf.logging.INFO)

    # recover variables from the input dict (config file)
    run_dir = hypes["train"]["run_dir"]
    log_dir = hypes["train"]["log_dir"]
    ckpt_dir = hypes["train"]["ckpt_dir"]
    pretrain_model = hypes["train"]["pretrain_model"]
    data_train = hypes["train"]["data_train"]
    data_val = hypes["train"]["data_val"]
    learning_rate = hypes["train"]["learning_rate"]
    export_model = hypes["train"]["export_model"]
    is_tfrecord = hypes["train"].get("tfrecord_flag", False)
    batch_size = hypes["train"].get("batch_size", 32)

    # 数据增强的参数: 获取默认参数，或者从配置文件中加载
    augment_params = hypes["train"].get("augment", get_default_augment_params())

    if is_tfrecord:
        _num_classes = hypes['train'].get('num_classes', None)
        if _num_classes is None:
            raise ValueError("Missing num_classes setting in the config file when training data is tfrecord files.")

        img_data = ImageData(data_train, data_val,
                             is_tfrecord=is_tfrecord,
                             batch_size=batch_size,
                             augment_params=augment_params,
                             output_like='Inception')

    else:
        # 如果只想在部分类别熵测试，可以在配置文件中指定 use_classes
        # 例如 "use_classes": ["000101", "000011"]
        # 即只使用上面两种类别的数据训练和验证
        # 如果 "use_classes"为[]或不存在，默认使用全部类别
        small_classes_set = None
        _classes_set = hypes['train'].get('use_classes', None)
        if _classes_set:
            small_classes_set = _classes_set

        img_data = ImageData(data_train, data_val,
                             small_classes_set=small_classes_set,
                             batch_size=batch_size,
                             augment_params=augment_params,
                             output_like='Inception')
        _num_classes = len(img_data.classes)


    # 网络相关的配置参数
    net_params = {
        'num_classes': _num_classes,
        'pretrain_model': pretrain_model,
        'checkpoint_path': ckpt_dir,
        'exclude': ['logits'],
        'adam_beta1': hypes_train['adam_beta1']
    }

    tf.logging.info('Train on {} classes.'.format(net_params['num_classes']))

    graph = tf.Graph()
    with graph.as_default():
        train_dataset = img_data.data_input_fn(mode='train')
        val_dataset = img_data.data_input_fn(mode='eval')
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_initializable_iterator()
        val_iterator = val_dataset.make_initializable_iterator()
        images, labels = iterator.get_next()

        net = build_net(images, labels, net_params)
        net['handle'] = handle
        saver = tf.train.Saver()

    # Indicates whether we are in training or in test mode
    is_training = net['is_training']

    # create writer to log the training; later for tensorboard
    train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), graph)
    eval_writer = tf.summary.FileWriter(os.path.join(log_dir, 'eval'))

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        net['train_handle'] = sess.run(train_iterator.string_handle())
        net['val_handle'] = sess.run(val_iterator.string_handle())
        # 如果有之前的checkpoint, 从其restore
        saved_ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if saved_ckpt:
            saver.restore(sess, saved_ckpt)

        # Here we initialize the iterator with the training set.
        # This means that we can go through an entire epoch until the iterator becomes empty.
        train_and_evaluate(sess, train_iterator, val_iterator,
                           train_writer, eval_writer, net,
                           saver, hypes, float(learning_rate))
        tf.logging.info('Training finished, evaluating the model on validation set...')
        eval_result = fully_evaluate(sess, val_iterator, net['eval_metric'],
                                     {is_training: False, handle: net['val_handle']}, -1)
        tf.logging.info('======= eval result ========')
        tf.logging.info(eval_result)
    export_pb_file(net_params, run_dir, export_model)



def get_default_augment_params():
    return {'horizontal_flip': True,
            'vertical_flip': True,
            'rotate': 180,
            'crop_probability': 0.5,
            'min_crop_percent': 0.5,
            'max_crop_percent': 1.0}


if __name__ == '__main__':

    import argparse
    import json

    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('--config', action="store", type=str, default='config/config.json')
    parser.add_argument('--run_dir', action="store", type=str, default='RUN')
    args = parser.parse_args()

    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    assert (os.path.exists(args.run_dir) and os.path.isdir(args.run_dir))
    assert (os.path.exists(args.config) and not os.path.isdir(args.config))
    hypes = json.load(open(args.config, 'r'))
    hypes_train = hypes['train']
    project_name = hypes_train['project_name']

    hypes_train['run_dir'] = os.path.join(args.run_dir, project_name)
    hypes_train['log_dir'] = os.path.join(args.run_dir, project_name, 'logs')
    hypes_train['ckpt_dir'] = os.path.join(args.run_dir, project_name, 'checkpoints')
    mkdir_p(hypes_train['run_dir'])
    mkdir_p(hypes_train['log_dir'])
    mkdir_p(hypes_train['ckpt_dir'])

    # kick off training
    main(hypes)
