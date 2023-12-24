# coding: utf-8
# Copyright (C) 2021-2023, [Breezedeus](https://github.com/breezedeus).
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

import os
import sys
import logging

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnocr.utils import set_logger
from cnocr.dataset_utils import gen_dataset, collate_fn
from cnocr.data_utils.transforms import train_transform, test_transform

logger = set_logger(log_level=logging.INFO)


def test_gen_dataset():
    index_fp = 'data/test/train.tsv'
    img_folder = 'data/images'
    dataset = gen_dataset(index_fp, img_folder=img_folder, transforms=train_transform, mode='train')

    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4)
    for batch in iter(dataloader):
        print(batch)
        break
