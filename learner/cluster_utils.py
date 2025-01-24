"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-8


class KLDiv(nn.Module):
    def forward(self, predict, target):
        assert predict.ndimension() == 2, 'Input dimension must be 2'
        target = target.detach()
        p1 = predict + eps
        t1 = target + eps
        logI = p1.log()
        logT = t1.log()
        TlogTdI = target * (logT - logI)
        kld = TlogTdI.sum(1)
        return kld


class KCL(nn.Module):
    def __init__(self):
        super(KCL, self).__init__()
        self.kld = KLDiv()

    def forward(self, prob1, prob2):
        kld = self.kld(prob1, prob2)
        return kld.mean()


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
    return (weight.t() / torch.sum(weight, 1)).t()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, weight=None, epsilon: float = 0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def linear_combination(self, x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y

    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, self.weight, reduction=self.reduction)
        return self.linear_combination(loss / n, nll, self.epsilon)


class ClusterLossBoost(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, cluster_num):
        super(ClusterLossBoost, self).__init__()
        self.cluster_num = cluster_num

    def forward(self, c_j, pseudo_label_all, index):
        device = c_j.device
        pseudo_label = pseudo_label_all[index]  # 当前miniBatch的数据伪标签

        # 获取每个类别的权重
        pseudo_index_all = pseudo_label_all != -1
        pseudo_label_all = pseudo_label_all[pseudo_index_all]
        idx, counts = torch.unique(pseudo_label_all, return_counts=True)
        freq = pseudo_label_all.shape[0] / counts.float()
        weight = torch.ones(self.cluster_num).to(device)
        weight[idx] = freq

        # 构建自标签（self-label learning) 损失函数
        pseudo_index = pseudo_label != -1  # 这里需要更改！！！！！

        if pseudo_index.sum() > 0:
            criterion = LabelSmoothingCrossEntropy(weight=weight).to(device)  # 默认reduction是求平均
            # criterion = nn.CrossEntropyLoss(weight=weight).to(device)  # 默认reduction是求平均
            loss_ce = criterion(
                c_j[pseudo_index], pseudo_label[pseudo_index].to(device)
            )
        else:
            loss_ce = torch.tensor(0.0, requires_grad=True).to(device)
        return loss_ce
