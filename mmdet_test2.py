#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/31 11:06
# @Author  : zyf
# @File    : mmdet_test.py
# @Software: PyCharm
import os
import cv2
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config_file = 'configs/swin/cascade_rcnn_swin_base_sense.py'
root_path = '/media/n504/disk2/fk/sense/cryo/test/' #test路径
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'work_dirs/cascade_rcnn_swin_base_sense20220505/latest.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
filelist=os.listdir(root_path)
for item in filelist:
    img = root_path+item
    result = inference_detector(model, img)
    model.show_result(img,result,out_file='out/'+item)
    show_result_pyplot(model,img,result)
