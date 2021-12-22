from easydict import EasyDict as edict

model = edict(
    type='small',
    name='swin_small_patch4_window7_224.pth',
    swin=dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,

        pretrained='pretrained/swin_small_patch4_window7_224.pth',
    ),
    neck=dict(
        in_channels=[96, 192, 384, 768],
        out_channel=256,
    ),
)

train = edict(
    min_sizes=[[8, 16], [16, 32], [64, 128], [256, 512]],
    steps=[4, 8, 16, 32],
    variance=[0.1, 0.2],
    clip=False,
    loc_weight=2.0,
    gpu_train=True,
    batch_size=4,
    ngpu=1,
    epoch=100,
    decay1=190,
    decay2=220,
    image_size=640,
    pretrain=True,
)
