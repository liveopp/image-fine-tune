#!/usr/bin/python
# -*- coding: utf-8 -*-

import click
import os
from multiprocessing import Pool
import hashlib
import json
from glob import glob


def md5sum(filename):
    ''' generator md5 for the input file

    :param filename: file path in file system
    :return: md5 in hex
    '''
    with open(filename, "rb") as fd:
        fcont = fd.read()
    fmd5 = hashlib.md5(fcont)
    return fmd5.hexdigest()


def task_proc(d_files):
    ''' task to collection md5

    :param d_files: files path in file system
    :return: a dict with md5 and file path
    '''
    print("Spawn process...")
    md5s = {}
    for f in d_files:
        m = str(md5sum(f))
        if m in md5s:
            md5s[m].append(f)
            try:
                # os.system('rm {}'.format(f))
                print('rm {}'.format(f))
            except Exception as e:
                print(e)
        else:
            md5s[m] = []
            md5s[m].append(f)

    print("Work finished...")
    return md5s


def del_files(objs):
    ''' remove all the files with same key

    :param objs: dictory with md5 and files path
    :return:
    '''
    for k, v in objs.items():
        if len(v) > 1:
            target = ' '.join(v)
            os.system('rm {}'.format(target))
            print('rm {}'.format(target))

    return



@click.command()
@click.option('--root_dir', default='/path/to/folder', help='root path of image folder')
@click.option('--out_file', default='/path/to/folder/out.json', help='record md5 for all files')
@click.option('--nproc', default=4, type=int, help='num process')
def main(root_dir, nproc, out_file):

    def _all_img_get(root_dir, pattern):
        ''' collect all the files

        :param root_dir: base folder
        :param pattern: kind of the file
        :return: a list with all the files
        '''
        return [y for x in os.walk(root_dir) for y in glob(os.path.join(x[0], pattern))]

    all_files = _all_img_get(root_dir, '*.jpg')
    print('Found {} files in {} folder'.format(len(all_files), root_dir))

    tt_dp = len(all_files)
    items_per_proc = int(tt_dp / nproc)

    p = Pool()
    print("Handle duplicate files")

    p_idx = []

    # 创建进程池
    for idx in range(nproc):
        p_idx.append(p.apply_async(task_proc,
                                   args=(all_files[idx * items_per_proc: (idx + 1) * items_per_proc],)))
    # 处理列表的最后部分文件
    if tt_dp//nproc != 0:
        p_idx.append(p.apply_async(task_proc,
                                   args=(all_files[(idx+1) * items_per_proc:],)))

    p.close()
    p.join()

    # 结果保存到json文件
    out = {}
    for i, r in enumerate(p_idx):
        out.update(r.get())
    with open(out_file, 'w') as f:
        f.write(json.dumps(out))

    # 删除相同的文件
    del_files(out)


if __name__ == '__main__':
    main()
