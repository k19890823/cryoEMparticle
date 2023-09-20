_base_ = 'cascade_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
#model = dict(
#    roi_head=dict(
#        mask_head=dict(num_classes=8)))

data_root = '../acoustic_detection/'
# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('cube',
            'ball',
            'cylinder',
            'human body',
            'tyre',
            'circle cage',
            'square cage',
            'metal bucket',)
#data = dict(
#    train=dict(
#        img_prefix='../acoustic_detection/train/images/',
#        classes=classes,
#        ann_file='../acoustic_detection/acoustic_detection_train.json'),
#    val=dict(
#        img_prefix='../acoustic_detection/train/annotations/',
#        classes=classes,
#        ann_file='../acoustic_detection/acoustic_detection_val.json'),
#    test=dict(
#        img_prefix='../acoustic_detection/train/annotations/',
#        classes=classes,
#        ann_file='../acoustic_detection/acoustic_detection_val.json'))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'acoustic_detection_train.json',
        img_prefix=data_root + 'train/images'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'acoustic_detection_val.json',
        img_prefix=data_root + 'train/images'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'test_A/images'))

evaluation = dict(metric=['bbox',])

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './chechpoints/cascade_mask_rcnn_swin_small_patch4_window7.pth'