#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
在训练之前，检验数据集中的图片，去掉有问题的图片，尝试转换图片格式为RGB，并去掉图片中的EXIF信息。

Usage example:
        python imgutils.py --input_dir "/path/to/img" --pattern "*/*.jpg" --task "1-2-3" --nproc 2
"""
import os
import sys
import click
import piexif
from PIL import Image
import numpy as np
from multiprocessing import Pool


def task_proc(task_list, fpath_list):
    '''

    :param task_list: 任务列表
    :param fpath_list: 文件路径的列表
    :return:
    '''
    for img_path in fpath_list:
        # check if the image file is valid
        if '1' in task_list:
            try:
                im = Image.open(img_path)
            except:
                print('removing the corrupted image file: {}'.format(img_path))
                os.system('rm {}'.format(img_path))
            else:
                try:
                    im.load()
                except:
                    print('removing the corrupted image file: {}'.format(img_path))
                    os.system('rm {}'.format(img_path))

        # remove EXIF info from the img file
        if '2' in task_list:
            try:
                piexif.remove(img_path)
            except:
                print('piexif raise exceptions when processing {}'.format(img_path))

        # convert image to RGB and 3-channel mode
        if '3' in task_list:
            saving_flag = False

            pil_im = Image.open(img_path)
            if pil_im.mode != "RGB":
                pil_im = pil_im.convert("RGB")
                saving_flag = True

            im = np.array(pil_im)
            if len(im.shape) < 2:
                im = im[0]
                pil_im = Image.fromarray(im)
                pil_im.save(img_path, "JPEG")
            elif len(im.shape) == 3 and im.shape[2] >= 4:
                im = im[:, :, :3]
                pil_im = Image.fromarray(im)
                pil_im.save(img_path, "JPEG")
            else:
                if saving_flag:
                    pil_im.save(img_path, "JPEG")


@click.command()
@click.option('--input_dir', default='/path/to/folder', help='root path of image folder')
@click.option('--pattern', default='*.jpg', help='file search pattern for glob')
@click.option('--task', default='1-2-3', help='task pipeline')
@click.option('--nproc', default=4, type=int, help='num process')
def main(input_dir, pattern, task, nproc):
    from glob import glob
    from os import path

    all_files = glob(path.join(input_dir, pattern))
    all_files.sort()
    print('Found {} files in {} folder'.format(len(all_files), input_dir))

    task_list = str(task).split('-')
    tt_files = len(all_files)
    files_per_proc = int(tt_files / nproc)
    p = Pool()
    for idx in range(nproc):
        p.apply_async(task_proc, args=(task_list, all_files[idx * files_per_proc: (idx + 1) * files_per_proc],))

    p.close()
    p.join()


if __name__ == '__main__':
    main()
