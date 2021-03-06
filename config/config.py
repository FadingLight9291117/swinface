# config.py
from easydict import EasyDict as edict

cfg_tiny = dict(
    model=dict(
        type='tiny',
        name='swin_tiny_patch4_window7_224',
        drop_path_rate=0.2,
        swin=dict(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
        ),
        pretrained='pretrained/swin_tiny_patch4_window7_224.pth',
    ),
    min_sizes=[[8, 16], [16, 32], [64, 128], [256, 512]],
    steps=[4, 8, 16, 32],
    variance=[0.1, 0.2],
    clip=False,
    loc_weight=2.0,
    gpu_train=True,
    batch_size=8,
    ngpu=1,
    epoch=250,
    decay1=190,
    decay2=220,
    image_size=640,
    return_layers=dict(stage1=1, stage2=2, stage3=3),
    in_channel=32,
    out_channel=64,
    pretrain=True,
    pretrained='retrained/swin_tiny_patch4_window7_224.pth'
)

cfg_tiny = edict(cfg_tiny)

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 64,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}
