from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

from swin_transformer import SwinTransformer
from net import FPN, SSH


class ClassHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(in_channels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class SwinFace(nn.Module):
    def __init__(self,
                 phase='train',
                 model_cfg=None,
                 in_channels_list=None,
                 out_channels=64):
        super().__init__()
        self.phase = phase

        if model_cfg is not None:
            self.backbone = \
                SwinTransformer(embed_dim=model_cfg.model.swin.embed_dim,
                                depths=model_cfg.model.swin.depths,
                                num_heads=model_cfg.model.swin.num_heads,
                                window_size=model_cfg.model.swin.window_size,
                                ape=model_cfg.model.swin.ape,
                                drop_path_rate=model_cfg.model.swin.drop_path_rate,
                                patch_norm=model_cfg.model.swin.patch_norm,
                                use_checkpoint=model_cfg.model.swin.use_checkpoint)
            in_channels_list = model_cfg.model.neck.in_channels
        else:
            self.backbone = SwinTransformer()

        # load swin transformer pretrained weight
        if model_cfg.model.swin.pretrained:
            swin_pretrained = model_cfg.model.swin.pretrained
            self.backbone.init_weights(swin_pretrained)

        if in_channels_list is None:
            in_channels_list = [96, 192, 384, 768]

        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        self.ssh4 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=4, in_channels=out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=4, in_channels=out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=4, in_channels=out_channels)

    def forward(self, x):
        backbone_outs = self.backbone(x)
        fpn_out = self.fpn(backbone_outs)

        feature1 = self.ssh1(fpn_out[0])
        feature2 = self.ssh2(fpn_out[1])
        feature3 = self.ssh3(fpn_out[2])
        feature4 = self.ssh4(fpn_out[3])

        features = [feature1, feature2, feature3, feature4]

        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output

    @staticmethod
    def _make_class_head(fpn_num=3, in_channels=64, anchor_num=2):
        class_head = nn.ModuleList()
        for i in range(fpn_num):
            class_head.append(ClassHead(in_channels, anchor_num))
        return class_head

    @staticmethod
    def _make_bbox_head(fpn_num=3, in_channels=64, anchor_num=2):
        bbox_head = nn.ModuleList()
        for i in range(fpn_num):
            bbox_head.append(BboxHead(in_channels, anchor_num))
        return bbox_head

    @staticmethod
    def _make_landmark_head(fpn_num=3, in_channels=64, anchor_num=2):
        land_mark_head = nn.ModuleList()
        for i in range(fpn_num):
            land_mark_head.append(LandmarkHead(in_channels, anchor_num))
        return land_mark_head


if __name__ == '__main__':
    from config import cfg_tiny

    model = SwinFace(model_cfg=cfg_tiny).cuda()
    x = torch.ones(1, 3, 1640, 640).cuda()
    y = model(x)
    print([i.size() for i in y])
