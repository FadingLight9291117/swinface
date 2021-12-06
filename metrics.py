import torch


def box_iou(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    if type(boxes1) is not torch.Tensor:
        boxes1 = torch.tensor(boxes1)
    if type(boxes2) is not torch.Tensor:
        boxes2 = torch.tensor(boxes2)

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])).clamp(
        0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    if type(wh1) is not torch.Tensor:
        wh1 = torch.tensor(wh1)
    if type(wh2) is not torch.Tensor:
        wh2 = torch.tensor(wh2)
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def Tp(preds_boxes, gts_boxes, iou_thres=0.45, isBox=True):
    """
    返回预测正确的预测框数量
    Args:
        preds_boxes: (Tensor[N, 4)
        gts_boxes: (Tensor[N, 4])
        iou_thres: iou阈值默认0.45，表示iou大于阈值的图片才被认为是正值
        isBox: 默认True代表预测框格式为(x1, y1, x2, y2)，False代表预测框格式(x, y, w, h)
    """
    if isBox:
        iou_matrix = box_iou(preds_boxes, gts_boxes)
    else:
        iou_matrix = wh_iou(preds_boxes, gts_boxes)
    max_iou = iou_matrix.max(dim=0).values
    true_num = max_iou[max_iou > iou_thres].__len__()
    return true_num


def iou2number(preds, gts, iou_thres, isBox):
    tps = 0
    fps = 0
    for i in range(len(preds)):
        boxes1 = preds[i]
        boxes2 = gts[i]
        tp = Tp(boxes1, boxes2, iou_thres, isBox)
        tps += tp
        fps += len(boxes1) - tp
    gt_num = 0
    for i1 in gts:
        for i2 in i1:
            for _ in i2:
                gt_num += 1
    fns = gt_num - tps
    return tps, fps, fns


def precision(preds, gts, iou_thres, isBox):
    tps, fps, fns = iou2number(preds, gts, iou_thres, isBox)
    p = tps / (tps + fps)
    return number_formatter(p)


def recall(preds, gts, iou_thres, isBox):
    tps, fps, fns = iou2number(preds, gts, iou_thres, isBox)
    r = tps / (tps + fns)
    return number_formatter(r)


def f1(preds, gts, iou_thres, isBox):
    tps, fps, fns = iou2number(preds, gts, iou_thres, isBox)
    p = tps / (tps + fps)
    r = tps / (tps + fns)
    f = 2 * p * r / (p + r)
    return number_formatter(f)


def p_r_f1(preds, gts, iou_thres, isBox):
    tps, fps, fns = iou2number(preds, gts, iou_thres, isBox)
    p = tps / (tps + fps)
    r = tps / (tps + fns)
    f = 2 * p * r / (p + r)
    return number_formatter(p), number_formatter(r), number_formatter(f)


def number_formatter(number):
    return f'{number:.4f}'


if __name__ == '__main__':
    preds_boxes = [[
        [0, 0, 13, 13],
        [10, 10, 23, 45],
        [1, 2, 45, 70]
    ]]
    gts_boxes = [[
        [0, 0, 13, 13],
        [10, 10, 20, 40]
    ]]

    print(p_r_f1(preds_boxes, gts_boxes, 0.45, True))
