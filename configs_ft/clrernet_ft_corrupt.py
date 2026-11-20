# Fine-tune CORRUPT: identical to clrernet_ft_control.py except one added corruption step in the train pipeline.
_base_ = ['./clrernet_ft_control.py']
compose_cfg = dict(bboxes=False, keypoints=True, masks=True)
crop_bbox = [0, 270, 1640, 590]
img_scale = (800, 320)
# verbatim copy of the authors' train_al_pipeline / train_pipeline with the corruption step inserted:
train_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    # ADDED (only change vs control): one corruption at full 1640x590 resolution, p=0.5.
    # albumentations' own implementations, NOT the test-time corruption code. Shadow is held out (unseen at test).
    dict(type="OneOf", transforms=[
        dict(type="RandomRain", slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, blur_value=3, brightness_coefficient=0.8, p=1.0),
        dict(type="MotionBlur", blur_limit=(7, 25), p=1.0),
        dict(type="ImageCompression", quality_lower=5, quality_upper=40, p=1.0),
        dict(type="RandomBrightnessContrast", brightness_limit=(-0.7, -0.2), contrast_limit=0.0, p=1.0),
    ], p=0.5),
    dict(
        type="Crop",
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
    dict(type="HorizontalFlip", p=0.5),
    dict(type="ChannelShuffle", p=0.1),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=0.04,
        contrast_limit=0.15,
        p=0.6,
    ),
    dict(
        type="HueSaturationValue",
        hue_shift_limit=(-10, 10),
        sat_shift_limit=(-10, 10),
        val_shift_limit=(-10, 10),
        p=0.7,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="MotionBlur", blur_limit=5, p=1.0),
            dict(type="MedianBlur", blur_limit=5, p=1.0),
        ],
        p=0.2,
    ),
    dict(
        type="IAAAffine",
        scale=(0.8, 1.2),
        rotate=(-10.0, 10.0),  # this sometimes breaks lane sorting
        translate_percent=0.1,
        p=0.7,
    ),
    dict(type="Resize", height=img_scale[1], width=img_scale[0], p=1),
]

train_pipeline = [
    dict(type="albumentation", pipelines=train_al_pipeline),
    #dict(type="Normalize", **img_norm_cfg),
    dict(
        type="PackCLRNetInputs",
        #keys=["img"],
        meta_keys=[
            "filename",
            "sub_img_name",
            "ori_shape",
            "img_shape",
            "ori_shape",
            "img_shape",
            "gt_points",
            "gt_masks",
            "lanes",
        ],
    ),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
