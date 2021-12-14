import os
import argparse
import time
import datetime
import math
import logging
from pathlib import Path

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
from data import WiderFaceDataset, detection_collate, preproc, cfg
from swinFace import SwinFace
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.timer import Timer
from utils.box_utils import decode, decode_landm
import metrics

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/', help='Training dataset directory')
parser.add_argument('--evaling_dataset', default='./data/widerface/val/', help='Evaling dataset directory')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./result/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
evaling_dataset = args.evaling_dataset
save_folder = args.save_folder

# config log =============================================================
now = datetime.datetime.now()
save_folder = str(Path(save_folder) / f'{now.year}{now.month}{now.day}{now.hour}{now.minute}')
Path(save_folder).mkdir(exist_ok=True)
log_path = f'{save_folder}/train.log'

# tensorboard config =========================================================
tb_writer = SummaryWriter(Path(save_folder).joinpath('log'))

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

model = SwinFace()
logger.debug("Printing net...")
logger.debug(model)

# Resume =================================================================
if args.resume_net is not None:
    logger.debug('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
# ========================================================================

if num_gpu > 1 and gpu_train:
    model = torch.nn.DataParallel(model).cuda()
else:
    model = model.cuda()

cudnn.benchmark = False

optimizer = optim.SGD(model.parameters(), lr=initial_lr,
                      momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
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

    # =============== prepare eval dataset ================================

    eval_dataset = WiderFaceDataset(evaling_dataset, preproc=eval_transform, dataset_type='eval')
    eval_dataloader = data.DataLoader(eval_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=detection_collate)
    # =======================================================================

    epoch_begin = 0 + args.resume_epoch
    epoch_size = math.ceil(len(train_dataset) / batch_size)

    start_iter = args.resume_epoch * epoch_size if args.resume_epoch > 0 else 0
    max_iter = max_epoch * epoch_size

    iter_num = start_iter
    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    scaler = amp.GradScaler()
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
                loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        
            # backprop
            
            # loss.backward()
            scaler.scale(loss/grad_accu_step).backward()
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

            if (epoch + 1) % 5 == 0 or ((epoch + 1) % 5 == 0 and (epoch + 1) > cfg['decay1']):
                # eval =====================================================================
                # info = evaluate(model, eval_dataset, device)
                # logger.debug(f'eval: {info}')
                # ==========================================================================
                save_path = Path(save_folder).joinpath('weights', f'swin_epoch_{epoch + 1}.pth')
                save_path.parent.mkdir(exist_ok=True)
                torch.save(model.state_dict(), save_path.__str__())

    save_path = Path(save_folder).joinpath('weights', f'swin_Final.pth')
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path.__str__())
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')

    # ======================================================================================================


@torch.no_grad()
def eval(model, eval_dataset, device):
    model.eval()
    model = model.to(device)
    for i, (images, targets) in enumerate(eval_dataset):
        img_h, img_w = images[0].size()[2:4]
        scale = torch.Tensor([img_w, img_h, img_w, img_h])
        images[:, 0, :, :] -= rgb_mean[0]
        images[:, 1, :, :] -= rgb_mean[1]
        images[:, 2, :, :] -= rgb_mean[2]

        images = images.to(device)
        scale = scale.to(device)

        locs, confs, _ = model(images)  # forward pass

        priorbox = PriorBox(cfg, image_size=(img_h, img_w))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data

        preds = []
        for loc, conf in zip(locs, confs):
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img_w, img_h] * 5)
            scale1 = scale1.to(device)
            landms = landms * scale1
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            preds.append(dets)


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
    # evaluate(model, eval_dataset, device)

    # model = SwinFace()
    # model.load_state_dict(torch.load('./result/2021124125/weights/swin_epoch_35.pth'))
