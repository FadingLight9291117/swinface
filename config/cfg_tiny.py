from easydict import EasyDict as edict

model = edict(
    type='tiny',
    name='swin_tiny_patch4_window7_224',
    swin=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,

        pretrained='pretrained/swin_tiny_patch4_window7_224.pth',
    ),
    neck=dict(in_channels=[96, 192, 384, 768])
)

min_sizes = [[8, 16], [16, 32], [64, 128], [256, 512]],
steps = [4, 8, 16, 32],
variance = [0.1, 0.2],
clip = False,
loc_weight = 2.0,
gpu_train = True,
batch_size = 8,
ngpu = 1,
epoch = 250,
decay1 = 190,
decay2 = 220,
image_size = 640,
return_layers = edict(stage1=1, stage2=2, stage3=3),
in_channel = 32,
out_channel = 64,
pretrain = True,
