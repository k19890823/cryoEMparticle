#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/30 10:38
# @Author  : zyf
# @File    : submit.py
# @Software: PyCharm
import json
import csv

"""
   2021年全国水下机器人大赛--水下声学目标检测赛项
   结果提交格式转化json->csv 
"""
labels = {
    1: "cube",  # 正方体
    2: "ball",  # 球体
    3: "cylinder",  # 圆柱体
    4: "human body",  # 人体模型
    5: "tyre",  # 轮胎
    6: "circle cage",  # 圆形地笼
    7: "square cage",  # 方形地笼
    8: "metal bucket",  # 铁桶
}

# 读取保存好的json文件
read_file = open('cascade_swin_rcnn_b_sonar_test_5_7.bbox.json', 'r')
read_json = json.load(read_file)
read_json_sort = sorted(read_json, key=lambda keys: int(keys['image_id']))
print(read_json_sort)
image_ids = []
rows = []
for item in read_json_sort:
    id = item['image_id']
    confidence = item['score']
    name = labels[item['category_id']]
    bbox = item['bbox']
    x_min = round(bbox[0])
    y_min = round(bbox[1])
    box_width = bbox[2]
    box_height = bbox[3]
    x_max = round(bbox[0] + bbox[2])
    y_max = round(bbox[1] + bbox[3])
    print((name, id, confidence, x_min, y_min, x_max, y_max))
    rows.append((name, id, confidence, x_min, y_min, x_max, y_max))
    image_ids.append(item['image_id'])
print(rows)
# 标题
headers = ['name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
with open('submit_20210507.csv', 'w', newline='') as ff:
    csv_file = csv.writer(ff)
    csv_file.writerow(headers)
    csv_file.writerows(rows)

# 找到每个id对应的所有box
# image_ids_new = sorted(list(set(image_ids)))
# print(image_ids_new)
# print(len(image_ids_new))
# res_dict = {}
# for id in image_ids_new:
#     id_list = []
#     for item in read_json:
#         image_id = item['image_id']
#         if id == image_id:
#             id_list.append({
#                 'bbox':item['bbox'],
#                 'score':item['score'],
#                 'category':item['category_id']
#             })
#     res_dict[id] = id_list
#
# print(res_dict)
