_base_ = 'cascade_rcnn_swin_base_fish.py'

# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#    roi_head=dict(
#        mask_head=dict(num_classes=8)))

data_root = '../sea_fish/'
# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ("holothurian",  # 海参
           "echinus",  # 海胆
           "scallop",  # 扇贝
           "starfish",)  # 海星

model = dict(
    test_cfg = dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.3,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'seafish_train.json',
        img_prefix=data_root + 'train/images'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'seafish_val.json',
        img_prefix=data_root + 'train/images'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'seafish_test_b.json',
        img_prefix=data_root + 'test-B-image/'))

