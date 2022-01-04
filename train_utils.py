import torch
from torch import nn
import numpy as np


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=False).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=False).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


def compute_center_loss(features, centers, targets):

    # centerloss implementation from https://github.com/louis-she/center-loss.pytorch/blob/master/loss.py

    features = features.reshape(features.size(0), -1)
    target_centers = centers[targets]
    criterion = torch.nn.MSELoss()
    center_loss = criterion(features, target_centers)
    return center_loss


def get_center_delta(features, centers, targets, alpha, train_on_gpu=True):
    # implementation equation (4) in the center-loss paper
    features = features.reshape(features.size(0), -1)
    targets, indices = torch.sort(targets)
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = target_centers - features
    uni_targets, indices = torch.unique(
            targets.cpu(), sorted=True, return_inverse=True)

    if train_on_gpu:
        uni_targets = uni_targets.cuda()
        indices = indices.cuda()

        delta_centers = torch.zeros(
            uni_targets.size(0), delta_centers.size(1)
        ).cuda().index_add_(0, indices, delta_centers)
    else:
        delta_centers = torch.zeros(
            uni_targets.size(0), delta_centers.size(1)
        ).cpu().index_add_(0, indices, delta_centers)

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
    same_class_feature_count = torch.sum(targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

    delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
    result = torch.zeros_like(centers)
    result[uni_targets, :] = delta_centers
    return result


class MixUpLoss(nn.Module):
    """
    Mixup implementation heavily borrowed from https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py#L42
    Adapt the loss function `crit` to go with mixup.
    """

    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        self.reduction = reduction

    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output, target[:, 0].long()), self.crit(output, target[:, 1].long())
            d = loss1 * target[:, 2] + loss2 * (1 - target[:, 2])
        else:
            d = self.crit(output, target)
        if self.reduction == 'mean':
            return d.mean()
        elif self.reduction == 'sum':
            return d.sum()
        return d

    def get_old(self):
        if hasattr(self, 'old_crit'):
            return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit


def mixup_data(x, y, alpha=0.4):

    """
    Returns mixed inputs, pairs of targets, and lambda
    """

    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha, batch_size)
    # t = max(t, 1-t)
    lam = np.concatenate([lam[:, None], 1 - lam[:, None]], 1).max(1)
    # tensor and cuda version of lam
    lam = x.new(lam)

    shuffle = torch.randperm(batch_size).cuda()

    x1, y1 = x[shuffle], y[shuffle]
    # out_shape = [bs, 1, 1]
    out_shape = [lam.size(0)] + [1 for _ in range(len(x1.shape) - 1)]

    # [bs, temporal, sensor]
    mixed_x = (x * lam.view(out_shape) + x1 * (1 - lam).view(out_shape))
    # [bs, 3]
    y_a_y_b_lam = torch.cat([y[:, None].float(), y1[:, None].float(), lam[:, None].float()], 1)

    return mixed_x, y_a_y_b_lam
