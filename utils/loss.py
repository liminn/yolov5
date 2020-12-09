# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        
        
class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        
        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def compute_loss(p, targets, model):  # predictions, targets, model
    """
    Args:
        p: 网络输出，List[torch.tensor * 3], p[i].shape = (b, 3, h, w, nc+5), hw分别为特征图的长宽,b为batch-size
        targets: targets.shape = (nt, 6) , 6=icxywh,i表示第一张图片，c为类别，然后为坐标xywh
                xy是相对值么？ wh是相对值
        model: 模型

    Returns:

    """
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    # 获得标签分类，边框，索引，anchor
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
    
    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs
    # 设置三个特征图对应输出的损失系数
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        # 根据indices获取索引，方便找到对应网格的输出
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        # 置信度，shape (b, 3, h, w, 1)
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            # 找到对应网格的输出
            # ps shape (n, 85)
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            # 对输出xywh做反算
            # 其实这里的操作就是把xywh的logits，转换成预测值，在infernce时也这么写
            # pxy shape (n, 2)
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            # pwh shape (n, 2)
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            # pbox shape (n, 4)
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            # 计算边框损失，注意这个CIoU=True，计算的是ciou损失
            # 看giou，diou，ciou原理，看ciou代码
            # pbox.T shape: (4,n)  tbox[i] shape: (n,4)
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss
            
            # Objectness
            # 根据model.gr设置objectness的标签值
            # 这里model.gr=1，也就是说完全使用标签框与预测框的giou值来作为该预测框的objectness标签
            # iou 的shape是什么，如何赋值给tobj相应元素的？shape是如何对应上的
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
            
            # Classification
            # 设置如果类别数大于1才计算分类损失
            # 看一下上面的label smoothing， 理一下这儿的维度
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no == 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    # ai.shape = (na, nt) 生成anchor索引，每一列均为(0,1,2)
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    # targets shape 输入时是(nt,6)，处理后变为(na, nt, 7)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    # 设置偏移量
    g = 0.5  # bias
    # off shape (5,2)， 分别代表 相对于j(左上角x) k(左上角y) l(右下角x) m(右下角y)
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets
    
    # 对每个检测层进行处理
    for i in range(det.nl):
        # my note: det.anchors shape:(3,3,2),  anchors shape:(3,2)
        anchors = det.anchors[i]
        """
        p[i].shape = (b, 3, h, w，nc+5), hw分别为特征图的长宽
        gain = [1, 1, w, h, w, h, 1]
        """
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        # my note: transfer x/y/w/h 0~1 to feature map size
        # 将标签框的xywh从基于0~1映射到基于特征图
        # t shape (na, nt, 7)
        t = targets * gain
        if nt:
            # my note: t[:, :, 4:6] shape (na, nt, 2), anchors[:, None] shape (na, 1, 2), so r shape is (na, nt, 2)
            # my q: why anchors[:, None] shape is (3, 1, 2)
            # 计算预测的wh与anchor的wh比值
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            # my note: torch.max(r, 1. / r) shape is (na, nt, 2) torch.max(r, 1. / r).max(2)[0] shape is (na, nt)
            # my note: j shape is (na, nt) type is bool 
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            # my note: t shape is (3*nt', 7) (m,7)
            # my q: why t become  (3*nt', 7)?
            # 筛选过后的t.shape = (M, 7),M为筛选过后的数量
            # 即筛选出属于该层的GTbox，正样本
            t = t[j]  # filter
            
            # Offsets
            # my note: gxy is the positive label center, shape is (m,2)
            # 得到GT的中心点坐标xy(相对于左上角的), (M, 2)
            gxy = t[:, 2:4]  # grid xy
            # my note: gain[[2, 3]] shape is (2,)
            # 得到中心点相对于右下角的坐标, (M, 2)
            gxi = gain[[2, 3]] - gxy  # inverse
            """
            把中心点相对于各个网格左上角x<0.5,y<0.5和相对于右下角的x<0.5,y<0.5的框提取出来；
            也就是j,k,l,m，在选取gij(也就是标签框分配给的网格的时候)对这四个部分的框都做一个偏移(减去上面的off),也就是下面的gij = (gxy - offsets).long()操作；
            再将这四个部分的框与原始的gxy拼接在一起，总共就是五个部分；
            也就是说：①将每个网格按照2x2分成四个部分，每个部分的框不仅采用当前网格的anchor进行回归，也采用该部分相邻的两个网格的anchor进行回归；
            原yolov3就仅仅采用当前网格的anchor进行回归；
            估计是用来缓解网格效应，但由于v5没发论文，所以也只是推测，yolov4也有相关解决网格效应的措施，是通过对sigmoid输出乘以一个大于1的系数；
            这也与yolov5新的边框回归公式相关；
            由于①，所以中心点回归也从yolov3的0~1的范围变成-0.5~1.5的范围；
            所以中心点回归的公式变为：
            xy.sigmoid() * 2. - 0.5 + cx
            """
            # 为什么要>1?
            # j/k shape: (1,m) j代表x相对于网格点左上角是否小于0.5，k代表y相对于网格点左上角是否小于0.5
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            # l/m shape: (1,m) l代表x相对于网格点右下角是否小于0.5，m代表y相对于网格点右下角是否小于0.5
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            # j shape (5,m)
            # 每一列均类似(1,0,1,1,0)，j和l互斥，k和m互斥
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            # t.repeat((5, 1, 1)) shape (5,m,7), final t shape (n,7)
            # 得到筛选的框(N, 7), N为筛选后的个数
            # 相当于人为增加了两倍的GT数量
            t = t.repeat((5, 1, 1))[j]
            # torch.zeros_like(gxy) shape (m,2) torch.zeros_like(gxy)[None] shape (1, m,2), why??
            # off shape (5,2) off[:, None] shape (5,1,2)
            # offsets shape (5,m,2) -> (n, 2)
            # 添加偏移量
            # (1, M, 2) + (5, 1, 2) = (5, M, 2) --[j]--> (N, 2)
            # 对每个GT，设置好相应的偏移量，其实就是四邻域里面进行偏移
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        # t[:, :2].long().T shape is (2,n), b/c shape is (n,)
        # b为batch中哪一张图片的索引，c为类别
        b, c = t[:, :2].long().T  # image, class
        # GTbox的中心点坐标，相对于feature_map的绝对值
        # gxy shape (n, 2)
        gxy = t[:, 2:4]  # grid xy
        # GTbox的宽高，相对于feature_map的绝对值
        # gwh shape (n, 2)
        gwh = t[:, 4:6]  # grid wh
        # 对应于原yolov3中，gij = gxy.long()
        # 对gxy进行相应的偏移
        # 通过.long取整，找出index
        # gij shape (n, 2)
        gij = (gxy - offsets).long()
        # gi/gj shape (n,)
        gi, gj = gij.T  # grid xy indices
        
        # Append
        # a为每个GT所对应的anchor的索引, shape (n,)
        a = t[:, 6].long()  # anchor indices
        # 添加索引，方便计算损失的时候取出对应位置的输出
        # b为batch中哪一张图片的索引，c为类别 (n)
        # a为每个GT所对应的anchor的索引, shape (n,)
        # gj为GT所属网格点左上角的y的值，shape (n,)
        # gi为GT所属网格点左上角的x的值，shape (n,)
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        # shape (n,4)
        # gxy - gij 为 每个GT相对于网格点左上角的偏移值 (n,2)
        # gwh 为每个GT的宽高，相对于feature_map的绝对值 (n,2)
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        # anchors[a] shape (n,2) 代表每个GT所对应的anchor的宽高大小，这里的宽高是相对于feature_map的绝对值
        anch.append(anchors[a])  # anchors
        # c shape (n, ) c代表每个GT所对应的类别
        tcls.append(c)  # class
        
    return tcls, tbox, indices, anch
