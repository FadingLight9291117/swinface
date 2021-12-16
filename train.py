import os
import argparse
import time
import datetime
import math
import logging
from pathlib import Path
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.ops import nms
import cv2
import numpy as np

from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from data import WiderFaceDataset, detection_collate, preproc
from config import cfg_tiny, cfg_small
from swinFace import SwinFace
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.timer import Timer
from utils.box_utils import decode, decode_landm
import metrics

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--network', default='small', help='')
parser.add_argument('--training-dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume-net', default=None, help='resume net for retraining')
parser.add_argument('--resume-epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--project', default='./runs/train_small', help='Location to save checkpoint models')
parser.add_argument('--name', default='exp', help='save to project/name')

args = parser.parse_args()

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
project = args.project
name = args.name

rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2
if args.network == 'tiny':
    cfg = cfg_tiny
elif args.network == 'small':
    cfg = cfg_small
else:
    msg = f'{args.network} 参数错误'
    raise ValueError(msg)

model_cfg = cfg.model
train_cfg = cfg.train
img_dim = train_cfg.image_size
num_gpu = train_cfg.ngpu
batch_size = train_cfg.batch_size
max_epoch = train_cfg.epoch
gpu_train = train_cfg.gpu_train

# config save_dir========================================================
Path(project).mkdir(exist_ok=True, parents=True)
save_dir = str(Path(project) / name)
i = 1
while os.path.exists(save_dir):
    i += 1
    save_dir = str(Path(project) / (name + f'{i}'))
Path(save_dir).mkdir()
# ======================================================================

# tensorboard log
tb_writer = SummaryWriter(os.path.join(save_dir, 'log'))

# config log =============================================================
log_path = f'{save_dir}/train.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
# ========================================================================

model = SwinFace(model_cfg=model_cfg)
logger.debug("Printing net...")
logger.debug(model)

if num_gpu > 1 and gpu_train:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()

cudnn.benchmark = True

optimizer = optim.SGD(model.parameters(), lr=initial_lr,
                      momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

scaler = amp.GradScaler()

# Resume =================================================================
if args.resume_net is not None:
    logger.debug('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    for key in state_dict.keys():
        if key == 'model':
            model.load_state_dict(state_dict[key])
        if key == 'optimizer':
            optimizer.load_state_dict(state_dict[key])
        if key == 'scaler':
            scaler.load_state_dict(state_dict[key])
        # if key == 'scheduler':
        #     scheduler.load_state_dict(state_dict[key])
# ========================================================================


priorbox = PriorBox(train_cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()


def eval_transform(img: np.ndarray, targ):
    img_h, img_w = img.shape[:2]
    img = cv2.resize(img, (img_dim, img_dim))
    img = img.transpose(2, 0, 1)
    bbox = targ[:4]
    landms = targ[4:-1]
    label = targ[-1]
    bbox[0::2] /= img_w
    bbox[1::2] /= img_h
    landms[0::2] /= img_w
    landms[1::2] /= img_h

    return img, targ


device = 'cuda'
grad_accu_step = 8


def train():
    logger.debug('Loading Dataset...')
    train_dataset = WiderFaceDataset(training_dataset, preproc(img_dim, rgb_mean))
    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                       collate_fn=detection_collate)

    epoch_begin = 0 + args.resume_epoch
    epoch_size = math.ceil(len(train_dataset) / batch_size)

    start_iter = args.resume_epoch * epoch_size if args.resume_epoch > 0 else 0
    max_iter = max_epoch * epoch_size

    iter_num = start_iter
    stepvalues = (train_cfg['decay1'] * epoch_size, train_cfg['decay2'] * epoch_size)
    step_index = 0

    # train ==================================================================================================
    for epoch in range(epoch_begin, max_epoch):
        for i, (images, targets) in enumerate(train_dataloader):
            # if i == 10: break # test
            model.train()
            t0 = time.time()

            # 根据迭代数调整学习率
            iter_num += 1
            if iter_num in stepvalues:
                step_index += 1
            lr = adjust_learning_rate(
                optimizer, gamma, epoch, step_index, iter_num, epoch_size)

            # load train data
            images, targets = images.to(device), [anno.to(device) for anno in targets]

            # forward
            with amp.autocast():
                out = model(images)
                loss_l, loss_c, loss_landm = criterion(out, priors, targets)
                loss = cfg_tiny.train['loc_weight'] * loss_l + loss_c + loss_landm

            # backprop
            # loss.backward()
            scaler.scale(loss / grad_accu_step).backward()
            # optimizer.step()
            if iter_num % grad_accu_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            t1 = time.time()
            # log
            batch_time = t1 - t0
            eta = int(batch_time * (max_iter - iter_num))

            info = {
                'Epoch': f'{epoch}/{max_epoch}',
                'step': f'{i + 1}/{epoch_size}',
                'Iter': f'{iter_num}/{max_iter}',
                'loss': {
                    'Loc': float(f'{loss_l.item():.4f}'),
                    'Cls': float(f'{loss_c.item():.4f}'),
                    'Landm': float(f'{loss_landm.item():.4f}'),
                },
                'LR': float(f'{lr:.8f}'),
                'Batchtime': float(f'{batch_time:.4f}'),
                'ETA': str(datetime.timedelta(seconds=eta)),
            }
            logger.debug('train: ' + str(info))

            tb_writer.add_scalar('loc_loss', info['loss']['Loc'], iter_num, walltime=None)
            tb_writer.add_scalar('cls_loss', info['loss']['Cls'], iter_num, walltime=None)
            tb_writer.add_scalar('landm_loss', info['loss']['Landm'], iter_num, walltime=None)

            if (epoch + 1) % 5 == 0 or ((epoch + 1) % 5 == 0 and (epoch + 1) > cfg_tiny['decay1']):
                # eval =====================================================================
                # info = evaluate(model, eval_dataset, device)
                # logger.debug(f'eval: {info}')
                # ==========================================================================
                save_path = Path(save_dir).joinpath('weights', f'swin_epoch_{epoch + 1}.pth')
                save_path.parent.mkdir(exist_ok=True)
                save_state(save_path, model=model, optimizer=optimizer, scaler=scaler)

    save_path = Path(save_dir).joinpath('weights', f'swin_Final.pth')
    save_path.parent.mkdir(exist_ok=True)
    save_state(save_path, model=model, optimizer=optimizer, scaler=scaler)


def save_state(save_path, model=None, optimizer=None, scheduler=None, scaler=None):
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(exist_ok=True, parents=True)

    status = OrderedDict(model=model.state_dict(),
                         optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict())

    torch.save(status, save_path)


# sche = optim.lr_scheduler.LambdaLR(optimizer, lambda x: lr)
# sche.step()
# sche.state_dict()

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
