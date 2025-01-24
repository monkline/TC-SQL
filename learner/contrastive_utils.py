"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 12/12/2021
"""


from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np


class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05, m=0.4):
        super(PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        self.m = m
        print(f"\n Initializing PairConLoss \n")

    def forward(self, features_1, features_2):
        loss = self.MCL(features_1, features_2, self.m, self.temperature)
        return {"loss":loss}

    def MCL(self, feature1, feature2, m, t):
        cos_theta12 = torch.matmul(feature1, feature2.T)  # N * N
        cos_theta_diag12 = torch.diag(cos_theta12)  # ii: N * 1
        theta12 = torch.acos(cos_theta_diag12)
        cos_theta_diag_m12 = torch.cos(theta12 + m)  # ii：positve pairs N * 1
        cos_theta11 = torch.matmul(feature1, feature1.T)  # 对角线元素都不用，其他的作为负样本
        cos_theta22 = torch.matmul(feature2, feature2.T)
        cos_theta21 = cos_theta12.T
        neg12 = torch.sum(torch.exp(cos_theta12 / t), dim=1) - torch.exp(cos_theta_diag12 / t)
        neg11 = torch.sum(torch.exp(cos_theta11 / t), dim=1) - torch.exp(torch.diag(cos_theta11) / t)
        neg22 = torch.sum(torch.exp(cos_theta22 / t), dim=1) - torch.exp(torch.diag(cos_theta22) / t)
        neg21 = torch.sum(torch.exp(cos_theta21 / t), dim=1) - torch.exp(torch.diag(cos_theta21) / t)
        cml_loss = -torch.mean(torch.log(torch.exp(cos_theta_diag_m12 / t) / (neg12 + neg11 + torch.exp(cos_theta_diag_m12 / t)))) - torch.mean(torch.log(torch.exp(cos_theta_diag_m12 / t) / (neg21 + neg22 + torch.exp(cos_theta_diag_m12 / t))))
        return cml_loss


class InstanceLossBoost(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, temperature, cluster_num, m=0):
        super(InstanceLossBoost, self).__init__()
        self.temperature = temperature
        self.cluster_num = cluster_num
        self.m = m


    def forward(self, z_i, z_j, pseudo_label):
        device = z_i.device
        n = z_i.shape[0]
        invalid_index = pseudo_label == -1  # 没有为标签的数据的索引
        
        mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to(device )
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(n).float().to(device)
        mask &= ~(mask_eye.bool())
        mask = mask.float()
        
        mask = mask.repeat(2, 2)
        mask_eye = mask_eye.repeat(2, 2)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(n*2).view(-1, 1).to(device),
            0,
        )
        logits_mask *= 1 - mask  # 负例对
        mask_eye = mask_eye * logits_mask  # 正例对

        z = torch.cat((z_i, z_j), dim=0) 
        sim = torch.matmul(z, z.t()) / self.temperature  # z @ z.t() / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)  # 获取每一行的最大值, 并保持2*n行1列
        sim = sim - sim_max.detach()  #  这样做是为了防止上溢，因为后面要进行指数运算

        exp_sim_neg = torch.exp(sim) * logits_mask  # 得到只有负例相似对的矩阵
        log_sim = sim - torch.log(exp_sim_neg.sum(1, keepdim=True))  #  log_softmax(), 分子上 正负例对 都有

        # compute mean of log-likelihood over positive
        instance_loss = -(mask_eye * log_sim).sum(1) / mask_eye.sum(1)  # 去分子为正例对的数据 
        instance_loss = instance_loss.view(2, n).mean()

        return {"loss": instance_loss}


