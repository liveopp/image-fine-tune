#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
从一个数据集中分一部分图片出来作为validation set.
数据源的根目录和目标目录应有相同的子目录，以及目录结构，例如:

    000001
    000002
    .....
    000100

Usage example:
        python dataset_split.py --src_root "/path/to/dataset_root" --dst_root "/path/to/val_root" --ratio 0.1
"""
import sys
import os
from os import path
from glob import glob
from random import shuffle
import click

@click.command()
@click.option('--src_root', default='/path/to/folder', help='root path of train')
@click.option('--dst_root', default='/path/to/folder', help='root path of val')
@click.option('--ratio', default=0.05, help='percentage of val images to split from the input dataset')
def main(src_root, dst_root, ratio):
    assert (path.exists(src_root))
    assert (path.exists(dst_root))

    split_ratio = ratio
    all_img_path = glob(path.join(src_root, "*/*.jpg"))
    shuffle(all_img_path)
    val_img_path = all_img_path[:int(len(all_img_path) * split_ratio)]
    print('{} images, will move {} images to val set. split ratio: {}'.format(len(all_img_path), len(val_img_path),
                                                                              split_ratio))

    for _img_path in val_img_path:
        sub_dir = _img_path.split('/')[-2]
        dst = path.join(dst_root, sub_dir)
        os.system('mv {} {}'.format(_img_path, dst))


if __name__ == '__main__':
    main()
