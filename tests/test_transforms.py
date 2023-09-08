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
import os

import torch
from PIL import Image
import cv2
import numpy as np
import albumentations as alb
import matplotlib.pyplot as plt

from cnocr import read_img
from cnocr.data_utils.transforms import CustomRandomCrop, _train_alb_transform, transform_wrap


def test_custom_random_crop():
    # 创建一个简单的测试图像
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.circle(test_image, (50, 50), 40, (255, 0, 0), -1)  # 在中心画一个蓝色的圆

    # 使用CustomRandomCrop变换
    transform = alb.Compose([
        CustomRandomCrop(crop_size=(10, 10))
    ])

    # 应用变换
    transformed_image = transform(image=test_image)["image"]
    assert transformed_image.shape == (100, 100, 3)

    # 显示原始和变换后的图像
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(test_image)
    ax[0].set_title("Original Image")
    ax[1].imshow(transformed_image)
    ax[1].set_title("Transformed Image")
    plt.savefig("test_custom_random_crop.png")


_train_alb_transform.transforms = _train_alb_transform.transforms[:-1]
train_transform = transform_wrap(_train_alb_transform)


def transform_alot(image_fp, out_fp):
    # image_fp = "examples/colorful.jpg"
    pillow_image = Image.open(image_fp).convert("RGB")
    image = read_img(image_fp).transpose((2, 0, 1))  # res: [1, H, W]
    h, w = image.shape[1:]
    image = torch.from_numpy(image)

    rows = 12
    cols = 4
    max_h, max_w = h, w

    all_images = [pillow_image]
    for i in range(rows * cols - 1):
        transformed_image = train_transform(image=image)  # -> [C, H, W]
        transformed_image = transformed_image.numpy()
        max_h = max(max_h, transformed_image.shape[1])
        max_w = max(max_w, transformed_image.shape[2])
        transformed_image = Image.fromarray(transformed_image.squeeze(0))
        all_images.append(transformed_image)

    # 创建一个新的空白图片，尺寸是原始图片宽度的4倍，高度的8倍
    result_image = Image.new("RGB", (max_w * cols, max_h * rows))

    # 循环调用transform函数31次，并将结果复制到结果图片中
    for i in range(rows * cols):
        # 计算当前图片在结果图片中的位置
        x = i % cols * max_w
        y = i // cols * max_h
        # 将变换后的图片复制到结果图片的相应位置
        result_image.paste(all_images[i], (x, y))

    # 保存结果图片
    result_image.save(out_fp)


def test_train_transform():
    image_fps = [
        '/Users/king/Documents/WhatIHaveDone/Test/text_renderer/output/number-pure/00000001.jpg',
        '/Users/king/Documents/WhatIHaveDone/Test/text_renderer/output/number-pure/00000007.jpg',
        '/Users/king/Documents/WhatIHaveDone/Test/text_renderer/output/number-pure/00000009.jpg',
        '/Users/king/Documents/WhatIHaveDone/Test/text_renderer/output/number-pure/00000015.jpg',
        '/Users/king/Documents/WhatIHaveDone/Test/text_renderer/output/number-pure/00000017.jpg',
        '/Users/king/Documents/WhatIHaveDone/Test/text_renderer/output/number-pure/00000025.jpg',
        '/Users/king/Documents/WhatIHaveDone/Test/text_renderer/output/number-pure/00000028.jpg',
        '/Users/king/Documents/WhatIHaveDone/Test/text_renderer/output/number-pure/00000193.jpg',
        '/Users/king/Documents/WhatIHaveDone/Test/text_renderer/output/number-pure/00000198.jpg',
    ]
    os.makedirs("test-out", exist_ok=True)
    for image_fp in image_fps:
        out_fp = os.path.basename(image_fp)
        out_fp = os.path.join("test-out", out_fp)
        transform_alot(image_fp, out_fp)
