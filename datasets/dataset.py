# -*- coding:utf-8 -*-
"""
处理数据的相关工具
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import tensorflow as tf


class ImageData:
    def __init__(self, train_path_or_label_file,
                 eval_path_or_label_file,
                 is_tfrecord=False,
                 output_like='Inception',
                 small_classes_set=None,
                 image_size=(299, 299),
                 batch_size=32,
                 augment_params=None):
        """
        Args:
            train_path_or_label_file: 训练数据路径或label file，
                如果为路径，子目录格式为directory/label/*.(jpg/png),
                如果是label file, 每行代表一个文件，例如
                ./1479426526335413494.jpg, 01001
                逗号前为文件名，之后为类别
            eval_path_or_label_file: 验证数据路径或label file，格式同上
            output_like: 输出Tensor的数值范围类似那种网络，Inception/VGG
            is_tfrecord: 输入数据是否是tfrecord
            small_classes_set: 指定使用一部分的类别，如果为None则使用所有类别
            image_size: 输出的image长宽
            batch_size: 输入Tensor包含样本数
            augment_params: 数据增强参数
              horizontal_flip=False, vertical_flip=False, rotate=0 最大旋转角度
              crop_probability=0,  裁剪的概率 min_crop_percent=0.6,  最小裁剪保留比例
              max_crop_percent=1.) 最大裁剪保留比例
        """
        if not os.path.exists(train_path_or_label_file):
            raise FileNotFoundError('{} not exist'.format(train_path_or_label_file))
        if not os.path.exists(eval_path_or_label_file):
            raise FileNotFoundError('{} not exist'.format(eval_path_or_label_file))

        self.is_tfrecord = is_tfrecord
        if is_tfrecord:
            self.train_files = self.list_tfrecord(train_path_or_label_file)
            self.eval_files = self.list_tfrecord(eval_path_or_label_file)

            if len(self.train_files) == 0:
                raise FileNotFoundError('No tfrecord files in {}'.format(train_path_or_label_file))
            if len(self.eval_files) == 0:
                raise FileNotFoundError('No tfrecord files in {}'.format(eval_path_or_label_file))

        else:
            train_class = self._get_classes(train_path_or_label_file)

            if small_classes_set:
                class_set = set(small_classes_set) & train_class
            else:
                class_set = train_class

            self.classes = list(class_set)
            self.classes.sort()
            self.label2int = dict(zip(self.classes, range(len(class_set))))
            self.train_files = self.list_images(train_path_or_label_file)
            self.eval_files = self.list_images(eval_path_or_label_file)

        self.image_size = image_size
        self.batch_size = batch_size
        self.augment_params = augment_params
        self.output_like = output_like

    @staticmethod
    def _get_classes(path_or_label_file):
        ''' 从输入的数据目录或者文件中推测出种类。
        :param path_or_label_file: 数据所在路径或者label文件
        :return: 分类任务的种类数量
        '''
        if os.path.isdir(path_or_label_file):
            path = path_or_label_file
            classes = set([p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))])
        else:
            labels = []
            with open(path_or_label_file, 'r') as f:
                for line in f.readlines():
                    _image, _label = line.strip().split(',')
                    labels.append(_label.strip())
            classes = set(labels)
        return classes

    def list_tfrecord(self, path):
        ''' 获取tfrecord文件路径的列表
        :param path: tfrecord文件所在的路径
        :return: tfrecord文件路径的列表
        '''
        assert os.path.isdir(path)
        files = []
        for f in os.listdir(path):
            if 'tfrecord' in f.lower():
                files.append(os.path.join(path, f))
        return files

    def list_images(self, path_or_label_file):
        """
        获取图片路径和label，
        """
        files, labels = [], []
        if os.path.isdir(path_or_label_file):
            directory = path_or_label_file
            labels_str = os.listdir(directory)
            for _label in labels_str:
                if _label in self.label2int:
                    for f in os.listdir(os.path.join(directory, _label)):
                        _f = f.lower()
                        if 'jpg' in _f or 'png' in _f or 'jpeg' in _f:
                            files.append(os.path.join(directory, _label, f))
                            labels.append(self.label2int[_label])
        else:
            base_path = '/'.join(path_or_label_file.split('/')[:-1]) \
                if '/' in path_or_label_file else './'
            with open(path_or_label_file, 'r') as f:
                for line in f.readlines():
                    _file, _label = line.strip().split(',')
                    _label = _label.strip()
                    if _label in self.label2int:
                        files.append(os.path.join(base_path, _file))
                        labels.append(self.label2int[_label])

        return files, labels

    @staticmethod
    def _augment_helper(images,
                        horizontal_flip=True,
                        vertical_flip=True,
                        rotate=90,  # 最大旋转角度
                        crop_probability=0.5,  # 裁剪的概率
                        min_crop_percent=0.6,  # Minimum linear dimension of a crop
                        max_crop_percent=1.):  # Maximum linear dimension of a crop

        with tf.name_scope('augmentation'):
            shp = tf.shape(images)
            batch_size, height, width = shp[0], shp[1], shp[2]
            width = tf.cast(width, tf.float32)
            height = tf.cast(height, tf.float32)

            # 生成所有需要对图片作用的变换
            # Every element is Nx8 tensor, where N is a batch size.
            transforms = []
            identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
            if horizontal_flip:
                # 有 50% 概率反转
                coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
                flip_transform = tf.convert_to_tensor(
                    [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
                transforms.append(
                    tf.where(coin,
                             tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                             tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

            if vertical_flip:
                coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
                flip_transform = tf.convert_to_tensor(
                    [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
                transforms.append(
                    tf.where(coin,
                             tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                             tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

            if rotate > 0:
                angle_rad = rotate / 180 * math.pi
                angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
                transforms.append(
                    tf.contrib.image.angles_to_projective_transforms(
                        angles, height, width))

            if crop_probability > 0:
                crop_pct = tf.random_uniform([batch_size], min_crop_percent,
                                             max_crop_percent)
                left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
                top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
                crop_transform = tf.stack([
                    crop_pct,
                    tf.zeros([batch_size]), top,
                    tf.zeros([batch_size]), crop_pct, left,
                    tf.zeros([batch_size]),
                    tf.zeros([batch_size])], 1)

                coin = tf.less(
                    tf.random_uniform([batch_size], 0, 1.0), crop_probability)
                transforms.append(
                    tf.where(coin, crop_transform,
                             tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

            if transforms:
                images = tf.contrib.image.transform(
                    images,
                    tf.contrib.image.compose_transforms(*transforms),
                    interpolation='BILINEAR')

        return images

    def _parse_image_file(self, filename, mode):
        ''' 加载图片文件，预处理，并返回 tensor

        :param filename: 图片文件的文件名
        :param mode: 'train' or 'eval'
        :return: a tensor object
        '''
        image_string = tf.read_file(filename)
        image = tf.image.decode_png(image_string, channels=3)
        if self.output_like == 'Inception':
            # convert to [0, 1]
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        elif self.output_like == 'VGG':
            image = tf.to_float(image)
        else:
            raise ValueError('Got incorrect value')

        if mode == 'train' and self.augment_params:
            image = tf.expand_dims(image, 0)
            image = self._augment_helper(image, **self.augment_params)
            image = tf.squeeze(image, [0])
        image_resize = tf.image.resize_images(image, self.image_size)

        return image_resize

    def _parse_tfrecord(self, serialized_example, mode):
        ''' tfrecord中序列化之后的对象做预处理，并转换成 tensor
        :param serialized_example: tfrecord中序列化之后的对象
        :param mode: 'train' or 'evla'
        :return:
        '''
        data = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature(shape=(), dtype=tf.string),
                'shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
                'label': tf.FixedLenFeature(shape=(), dtype=tf.int64),
            })

        image_raw = tf.decode_raw(data['image'], tf.uint8)
        image = tf.reshape(image_raw, data['shape'])
        if self.output_like == 'Inception':
            # convert to [0, 1]
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        elif self.output_like == 'VGG':
            image = tf.to_float(image)
        else:
            raise ValueError('Got incorrect value')

        if mode == 'train' and self.augment_params:
            image = tf.expand_dims(image, 0)
            image = self._augment_helper(image, **self.augment_params)
            image = tf.squeeze(image, [0])
        image_resize = tf.image.resize_images(image, self.image_size)
        image_resize = tf.reshape(image_resize, self.image_size+(3,))

        return image_resize, data['label']

    def data_input_fn(self, mode):
        """
        获取训练或者验证的dataset
        Args:
            mode: 'train' or 'eval'
        Returns:
            'tf.data.Dataset' 对象: 此dataset的每个元素都是 (features, labels)
            的tuple.
        """
        assert mode == 'train' or mode == 'eval'

        if self.is_tfrecord:
            dataset = tf.data.TFRecordDataset(self.train_files if mode == 'train' else self.eval_files)
            dataset = dataset.map(lambda _example: self._parse_tfrecord(_example, mode))
        else:
            img_files, labels = self.train_files if mode == 'train' else self.eval_files
            dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
            dataset = dataset.shuffle(buffer_size=len(img_files))
            dataset = dataset.map(lambda _img_file, _label: (self._parse_image_file(_img_file, mode), _label))

        dataset = dataset.batch(self.batch_size)

        return dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    augment_params = {
        'horizontal_flip': True,
        'vertical_flip': False,
        'rotate': 40,
        'crop_probability': 0.5,
        'min_crop_percent': 0.7,
        'max_crop_percent': 1.
    }
    img_data = ImageData('../data/train_label_file.csv', '../data/val',
                         augment_params=augment_params,
                         batch_size=1)
    dataset = img_data.data_input_fn(mode='train')
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    class_names = img_data.classes
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(10):
            try:
                img, label = sess.run(next_element)
                plt.imshow(img[0])
                plt.title('id: {}, label: {}, image shape: {}'.format(label[0],
                                                                      class_names[label[0]],
                                                                      img[0].shape))
                plt.show()
            except tf.errors.OutOfRangeError:
                break
