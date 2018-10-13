# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import random
import progressbar
import tensorflow as tf


def mkdir_p(path):
    """ make a folder in file system """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if os.path.isdir(path):
            pass
        else:
            raise


class Img2TFrecord(object):
    def __init__(self, path_or_label_file,
                 project_name='default',
                 small_classes_set=None,
                 example_per_file=1000):

        if not os.path.exists(path_or_label_file):
            raise FileNotFoundError('{} not exist'.format(path_or_label_file))

        classes = self._get_classes(path_or_label_file)
        if small_classes_set:
            class_set = set(small_classes_set) & classes
        else:
            class_set = classes
        self.classes = list(class_set)
        self.classes.sort()
        self.label2int = dict(zip(self.classes, range(len(class_set))))

        self.files_and_labels = self.list_images(path_or_label_file)
        self.example_per_file = example_per_file
        self.project_name = project_name

    @staticmethod
    def _get_classes(path_or_label_file):
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

    @staticmethod
    def image_file_to_tfexample(image_file, label):
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.convertScaleAbs(img)
        img_bytes = tf.compat.as_bytes(img.tostring())

        return tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=img.shape)),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }))

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

    def get_tfrecord_filename(self, base_path, shard_id):
        output_filename = '{}-{}.tfrecord'.format(self.project_name, shard_id)
        return os.path.join(base_path, output_filename)

    def dump_to_tfrecord(self, path):
        files, labels = self.files_and_labels
        assert len(files) == len(labels), 'The number of files should be equal with length of labels'

        file_and_label = list(zip(files, labels))
        random.shuffle(file_and_label)

        example_cnt = 0
        writer = None
        with progressbar.ProgressBar(max_value=len(files)) as bar:
            for _f, _l in file_and_label:
                if example_cnt % self.example_per_file == 0:
                    if writer:
                        writer.close()
                    writer = tf.python_io.TFRecordWriter(
                        self.get_tfrecord_filename(path, example_cnt // self.example_per_file))
                example = self.image_file_to_tfexample(_f, _l)
                writer.write(example.SerializeToString())
                bar.update(example_cnt)
                example_cnt += 1
            writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='将图片转化为TFRecord')
    parser.add_argument('--input', type=str, help='图片路径或label file')
    parser.add_argument('--output-path', type=str, dest='output_path',
                        help='TFRecord文件输出路径')
    parser.add_argument('--example-per-file', type=int,
                        default=1000, dest='example_per_file',
                        help='每个tfrecord文件包含样本数')
    parser.add_argument('--project-name', type=str, dest='project_name',
                        help='项目名(tfrecord文件名前缀)')

    args = parser.parse_args()

    img2tf = Img2TFrecord(args.input, args.project_name,
                          example_per_file=args.example_per_file)

    mkdir_p(args.output_path)
    img2tf.dump_to_tfrecord(args.output_path)
