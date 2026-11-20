# Fine-tune BROAD: identical to clrernet_ft_control.py except one added corruption step (broad, test-disjoint types).
_base_ = ['./clrernet_ft_control.py']
compose_cfg = dict(bboxes=False, keypoints=True, masks=True)
crop_bbox = [0, 270, 1640, 590]
img_scale = (800, 320)
# verbatim copy of the authors' train_al_pipeline / train_pipeline with the corruption step inserted:
train_al_pipeline = [
    dict(type="Compose", params=compose_cfg),
    # ADDED (only change vs control): one corruption at full 1640x590 resolution, p=0.5, from a BROAD set of corruption
    # types that are all DIFFERENT from the five test corruptions (darkening, motion blur, JPEG, cast shadow, rain).
    # Purpose: test whether diverse augmentation transfers to corruption types never seen in training (report section 7.1 found
    # no transfer from the narrow set to the held-out shadow). Note: the authors' recipe (all arms) already has mild
    # motion/median blur (<=5 px at 800x320, p=0.2) and small brightness/contrast/HSV jitter.
    dict(type="OneOf", transforms=[
        dict(type="RandomFog", fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.08, p=1.0),
        dict(type="RandomSnow", snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.0, p=1.0),
        dict(type="RandomSunFlare", flare_roi=(0, 0, 1, 0.5), src_radius=300, p=1.0),
        dict(type="GaussNoise", var_limit=(50.0, 400.0), p=1.0),
        dict(type="ISONoise", color_shift=(0.02, 0.08), intensity=(0.3, 0.8), p=1.0),
        dict(type="RandomContrast", limit=(-0.6, -0.2), p=1.0),
        dict(type="RGBShift", r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=1.0),
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
