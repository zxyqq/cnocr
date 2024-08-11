# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).
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

import base64
import io
from copy import deepcopy
from typing import List, Dict, Any

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile
from PIL import Image

from cnocr import CnOcr
from cnocr.utils import set_logger

from fastapi.middleware.cors import CORSMiddleware

logger = set_logger(log_level='DEBUG')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OCR_MODEL = CnOcr()
OCR_MODEL = CnOcr(cand_alphabet='0123456789')
# OCR_MODEL = CnOcr(det_model_name="en_PP-OCRv3_det", cand_alphabet='0123456789')
# OCR_MODEL = CnOcr(rec_model_name='number-densenet_lite_136-fc', det_model_name="en_PP-OCRv3_det", cand_alphabet='0123456789')
# OCR_MODEL = CnOcr(det_model_name='naive_det', cand_alphabet='0123456789')
# OCR_MODEL = CnOcr(rec_model_name='number-densenet_lite_136-fc-onnx', det_model_name='naive_det', cand_alphabet='0123456789')
# OCR_MODEL = CnOcr(rec_model_name='densenet_lite_136-gru', det_model_name='naive_det', cand_alphabet='0123456789')

class OcrResponse(BaseModel):
    status_code: int = 200
    results: List[Dict[str, Any]]

    def dict(self, **kwargs):
        the_dict = deepcopy(super().dict())
        return the_dict


@app.get("/")
async def root():
    return {"message": "Welcome to CnOCR Server!"}

class OcrRequest(BaseModel):
    imagebase64: str

@app.post("/ocr")
async def ocr(req: OcrRequest) -> Dict[str, Any]:
    imagebase64 = req.imagebase64
    if imagebase64.startswith('data:image/'):
        imagebase64 = imagebase64.split(',')[1]
    img_bytes = base64.b64decode(imagebase64)
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    res = OCR_MODEL.ocr(image)
    for _one in res:
        _one['position'] = _one['position'].tolist()
        if 'cropped_img' in _one:
            _one.pop('cropped_img')

    return OcrResponse(results=res).dict()
