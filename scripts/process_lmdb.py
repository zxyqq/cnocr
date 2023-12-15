# coding: utf-8
# Copyright (C) 2023, [Breezedeus](https://github.com/breezedeus).
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# !pip install lmdb

import sys
import os
import logging
import string

import click
import six
import lmdb
from PIL import Image
from torch.utils.data import Dataset


from cnocr import set_logger
from cnocr.consts import CN_VOCAB_FP
from cnocr.utils import read_charset

_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}
logger = set_logger(log_level=logging.INFO)

PUNCTUATIONS = "，。！？；：‘’“”（）【】《》「」『』、—…·-_,.!?;:'\"()[]{}<>\"\"/\\—…-"


def should_delete(img, labels, vocab):
    w, h = img.size
    if w < 6 or h < 6:
        return True
    if len(labels) < 1:
        return True
    if len(labels) == 1:
        if labels[0] in PUNCTUATIONS:
            return True
    # 可能是竖排文字
    if len(labels) > 1 and w < h:
        return True
    # 如果存在英文字母，则删除。（英文字母全部被变成了小写了，不准）
    if len(set(labels) & set(string.ascii_letters)) > 0:
        return True
    # 如果存在 label 不在 vocab 中，删除
    if len(set(labels).difference(vocab)) > 0:
        return True

    return False


class LmdbDataset(Dataset):
    """
    Ref: https://github.com/FudanVI/benchmarking-chinese-text-recognition/blob/main/data/lmdbReader.py
    """

    def __init__(self, root=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index > len(self):
            index = len(self) - 1
        assert index <= len(self), 'Error %d' % index
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf)
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))

            # if len(label) <= 0:
            #     return self[index + 1]
            #
            # label = label.lower()

        return (img, label)

    def dump_to_dir(self, output_dir, vocab, index_prefix_folder=''):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        suffix_dir = '/'.join(output_dir.split('/')[-3:])
        with open(os.path.join(output_dir, 'labels.txt'), 'w') as f:
            count = 0
            for index in range(len(self)):
                img, label = self[index]
                if should_delete(img, label, vocab):
                    continue
                label = label.replace('\t', ' ').replace('\n', '')
                label = list(label)
                label = [l if l != ' ' else '<space>' for l in label]
                img.save(os.path.join(output_dir, str(index) + '.jpg'))
                fp = os.path.join(index_prefix_folder, suffix_dir, str(index) + '.jpg')
                f.write('\t'.join([fp, ' '.join(label)]) + '\n')
                count += 1
        logger.info(f'Dumped {count} images to {output_dir}')


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('dump-lmdb')
@click.option(
    '-v', '--vocab-fp', type=str, default=CN_VOCAB_FP, help='字典文件路径',
)
@click.option(
    '-i', '--input-lmdb-dir', type=str, required=True, help='输入的 lmdb 文件夹路径',
)
@click.option(
    '--index-prefix-folder', type=str, default='', help='索引文件中记录的图片文件路径的前缀',
)
@click.option(
    '-o', '--output-dir', type=str, required=True, help='输出图片所在的文件夹',
)
def dump_lmdb(
    vocab_fp, input_lmdb_dir, index_prefix_folder, output_dir,
):
    """
    把 LMDB 数据集格式转换为 Huggingface Dataset 格式。
    """
    vocab, _ = read_charset(vocab_fp)
    lmdb_ds = LmdbDataset(input_lmdb_dir)
    lmdb_ds.dump_to_dir(
        output_dir, vocab=set(vocab), index_prefix_folder=index_prefix_folder
    )


if __name__ == "__main__":
    cli()
