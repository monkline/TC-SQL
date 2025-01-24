"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

from asyncore import write
import enum
from itertools import count
import os
from pickletools import optimize
from re import I
import time
import numpy as np
from sklearn import cluster
from sklearn.covariance import log_likelihood

from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader

import torch
import torch.nn as nn
from torch.nn import functional as F
from learner.cluster_utils import target_distribution, ClusterLossBoost
from learner.contrastive_utils import PairConLoss, InstanceLossBoost
# import ot
import sinkhornknopp as sk

import random
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args, scheduler=None):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta

        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.ce_loss = nn.CrossEntropyLoss()
        self.boost_ce_loss = ClusterLossBoost(cluster_num=self.args.num_classes)
        self.contrast_loss = PairConLoss(temperature=self.args.temperature, m=self.args.m)
        self.contrast_boost_loss = InstanceLossBoost(
            temperature=self.args.temperature, cluster_num=self.args.num_classes, m=self.args.m
            )

        N = len(self.train_loader.dataset)
        self.a = torch.full((N, 1), 1 / N).squeeze()

        self.b = torch.rand(self.args.classes, 1).squeeze()
        self.b = self.b / self.b.sum()

        self.u = None
        self.v = None
        self.h = torch.FloatTensor([1])
        self.allb = [[self.b[i].item()] for i in range(self.args.classes)]

        self.all_pseudo_labels = None  # 整体的数据点
        self.wo_untrust_labels = None  # 在 all pseudo albels 的基础上排除一些不可信的数据点

        print(f"*****Intialize SCCLv, temperature:{self.args.temperature}, eta:{self.args.eta}\n")

    def soft_ce_loss(self, pred, target, step):
        tmp = target ** 2 / torch.sum(target, dim=0)
        target = tmp / torch.sum(tmp, dim=1, keepdim=True)
        return torch.mean(-torch.sum(target * (F.log_softmax(pred, dim=1)), dim=1))

    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.args.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        return token_feat

    def prepare_transformer_input(self, batch):
        text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
        feat1 = self.get_batch_token(text1)
        if self.args.augtype == 'explicit':
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)
            input_ids = torch.cat(
                [feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)],
                dim=1
                )
            attention_mask = torch.cat(
                [feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1),
                 feat3['attention_mask'].unsqueeze(1)], dim=1
                )
        else:
            input_ids = feat1['input_ids']
            attention_mask = feat1['attention_mask']

        return input_ids.to(self.args.device), attention_mask.to(self.args.device)

    def loss_function(self, input_ids, attention_mask, selected, i):
        embd1, embd2, embd3 = self.model.get_embeddings(input_ids, attention_mask, task_type=self.args.augtype)

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat1, feat2)  # 对比损失
        # losses = self.contrast_boost_loss(feat1, feat2, self.wo_untrust_labels[selected])   # 去掉伪负例对 的 对比损失

        loss = self.eta * losses["loss"] / 2
        losses['contrast'] = losses["loss"]
        self.args.tensorboard.add_scalar('loss/contrast_loss', losses['loss'].item(), global_step=i)

        # Clustering loss
        if i >= self.args.pre_step + 1:
            P1 = self.model(embd1)
            P2 = self.model(embd2)
            P3 = self.model(embd3)  # predicted labels before softmax
            target_label = None
            if len(self.L.shape) != 1:
                if self.args.soft == True:
                    target = self.L[selected].to(self.args.device)
                    cluster_loss = self.soft_ce_loss(P2, target, i) + self.soft_ce_loss(P3, target, i)
                else:
                    target_label = torch.argmax(self.L, dim=1).to(self.args.device)
            else:
                target_label = self.L.to(self.args.device)
            if target_label != None:
                # print(P2.dtype, P3.dtype, target_label.dtype)
                target_label = target_label.to(torch.int64)
                # cluster_loss = self.ce_loss(P2, target_label[selected]) + self.ce_loss(P3, target_label[selected])
                cluster_loss = self.boost_ce_loss(P2, target_label, selected) + self.boost_ce_loss(
                    P3, target_label, selected
                    )
            loss += cluster_loss
            self.args.tensorboard.add_scalar('loss/cluster_loss', cluster_loss.item(), global_step=i)

            # 只训练cluster loss会帮助对比loss也变小，和有训练contrast loss时变化曲线相似
            # 只训练对比loss, cluster loss变小效果有限
            losses["cluster_loss"] = cluster_loss.item()

        losses['loss'] = loss
        self.args.tensorboard.add_scalar('loss/loss', loss, global_step=i)
        return loss, losses

    def train_step_explicit(self, input_ids, attention_mask, selected, i):
        if i >= self.optimize_times[-1]:
            _ = self.optimize_times.pop()
            if i >= self.args.pre_step + 40:
                self.get_labels(i)

        loss, losses = self.loss_function(input_ids, attention_mask, selected, i)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return losses

    def optimize_labels(self, step):
        # 1. aggregate P
        N = len(self.train_loader.dataset)
        PS = torch.zeros((N, self.args.classes))
        now = time.time()

        with torch.no_grad():
            for iter, (batch, selected) in enumerate(self.train_loader):
                input_ids, attention_mask = self.prepare_transformer_input(batch)
                emb1, _, _ = self.model.get_embeddings(
                    input_ids, attention_mask, task_type=self.args.augtype
                    )  # embedding
                p = F.softmax(self.model(emb1), dim=1)

                PS[selected] = p.detach().cpu()
                if iter == 0:
                    all_embeddings = emb1.detach()
                else:
                    all_embeddings = torch.cat((all_embeddings, emb1.detach()), dim=0)
        embeddings = all_embeddings.cpu().numpy()

        cost = -torch.log(PS)
        numItermax = 1000

        ###########
        if self.args.H == 'H2':
            # wang update b
            mu = 0.1
            z = torch.argmax(PS, dim=1)
            temp = list(range(self.args.num_classes))
            not_shown = list(set(temp).difference(set(z.numpy())))
            # print('not_shown ----:', len(not_shown))
            counts = Counter(z.numpy())
            for k in not_shown:
                counts[k] = 0
            self.b = mu * self.b + (1 - mu) * torch.tensor(list(counts.values())) / sum(counts.values())
        #############

        T, log = sk.sinkhorn_knopp(
            self.a, self.b, cost, self.args.epsion, numItermax=numItermax, warn=False, log=True, u=self.u, v=self.v,
            h=self.h, reg2=self.args.reg2, log_alpha=self.args.logalpha, Hy=self.args.H
            )
        # self.b = log['b']
        self.L = T
        self.wo_untrust_labels = self.get_pseudo_labels(embeddings, self.L, self.args.num_classes)
        print('Optimize Q takes {:.2f} min'.format((time.time() - now) / 60))

    def get_labels(self, step):
        # optimize labels
        print('[Step {}] Optimization starting'.format(step))
        # 更新self.L
        self.optimize_labels(step)

    def train(self):
        self.optimize_times = ((np.linspace(self.args.start, 1, self.args.M) ** 2)[::-1] * self.args.max_iter).tolist()
        # 训练前评估
        self.evaluate_embedding(-1)
        # for i in range(self.args.num_classes): self.optimize_times.pop()

        for i in np.arange(self.args.max_iter + 1):
            self.model.train()
            try:
                batch, selected = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch, selected = next(train_loader_iter)

            input_ids, attention_mask = self.prepare_transformer_input(batch)
            losses = self.train_step_explicit(input_ids, attention_mask, selected, i)

            if (self.args.print_freq > 0) and ((i % self.args.print_freq == 0) or (i == self.args.max_iter)):
                statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                flag = self.evaluate_embedding(i)
                if flag == -1:
                    break
        return None

    def evaluate_embedding(self, step):
        dataloader = unshuffle_loader(self.args)
        print('---- {} evaluation batches ----'.format(len(dataloader)))

        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label']
                # label -= 1
                feat = self.get_batch_token(text)
                embeddings = self.model.get_embeddings(
                    feat['input_ids'].to(self.args.device), feat['attention_mask'].to(self.args.device),
                    task_type="evaluate"
                    )
                pred = torch.argmax(self.model(embeddings), dim=1)

                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_pred = pred.detach()
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_pred = torch.cat((all_pred, pred.detach()), dim=0)

        # Initialize confusion matrices
        confusion = Confusion(max(self.args.num_classes, self.args.classes))
        rconfusion = Confusion(self.args.num_classes)
        embeddings = all_embeddings.cpu().numpy()
        pred_labels = all_pred.cpu()

        if step <= self.args.pre_step:
            if self.args.num_classes <= 20:
                cluster_model = cluster.KMeans(
                    n_clusters=self.args.classes,
                    init='k-means++',
                    n_init='auto',
                    random_state=self.args.seed, max_iter=3000, tol=0.01
                )
            else:
                cluster_model = cluster.AgglomerativeClustering(
                    n_clusters=self.args.classes,
                    metric="cosine",
                    linkage="average"
                )
            cluster_model.fit(embeddings)
            kpred_labels = torch.tensor(cluster_model.labels_.astype(np.int32))
            self.L = kpred_labels
            pred_labels = kpred_labels
            for i in range(self.b.shape[0]):
                self.b[i] = (self.L == i).sum() / self.L.shape[0]
        self.wo_untrust_labels = self.get_pseudo_labels(embeddings, pred_labels, self.args.num_classes)
        # clustering accuracy
        clusters_num = len(set(pred_labels.numpy()))
        print('preded classes number:', clusters_num)
        self.args.tensorboard.add_scalar('Test/preded_clusters', clusters_num, step)
        confusion.add(pred_labels, all_labels)
        _, _ = confusion.optimal_assignment(self.args.num_classes)

        acc = confusion.acc()
        clusterscores = confusion.clusterscores(all_labels, pred_labels)

        ressave = {"acc": acc}
        ressave.update(clusterscores)
        for key, val in ressave.items():
            self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)

        print('[Step]', step)
        print('[Model] Clustering scores:', clusterscores)
        print('[Model] ACC: {:.4f}'.format(acc))  # 使用kmeans聚类学到的representation的 acc

        # 停止标准：两次得到的label变化的占比少于tol
        y_pred = pred_labels.numpy()

        if step == -1:
            self.y_pred_last = np.copy(y_pred)
            self.wo_untrust_y_pred_last = np.copy(self.wo_untrust_labels)
        else:
            change_rate = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
            label_purity = self.label_purity(all_labels, y_pred)
            self.y_pred_last = np.copy(y_pred)
            self.wo_untrust_y_pred_last = np.copy(self.wo_untrust_labels)
            self.args.tensorboard.add_scalar('Test/change_rate', change_rate, step)
            self.args.tensorboard.add_scalar('Test/label_purity', label_purity, step)
            print('[Step] {} Label change rate: {:.3f} tol: {:.3f}'.format(step, change_rate, self.args.tol))
            print('[Step] {} Label purity: {:.3f} tol: {:.3f}'.format(step, label_purity, self.args.tol))
            if (step > self.args.pre_step and change_rate < self.args.tol) or step >= 3000:
                print('Reached tolerance threshold, stop training.')
                return -1
        # if step > 0 and change_rate > 0.97:
        #     self.args.pre_step = step

        # self.plot_(embeddings, all_labels-1, step, self.args.num_classes)
        return None

    def label_purity(self, true_labels, pred_labels):
        label_purity = []
        for i in range(self.args.num_classes):
            idx = true_labels == i
            subset_labels = pred_labels[idx]
            unique, counts = np.unique(subset_labels, return_counts=True)
            if counts.sum() > 0:
                label_purity.append(counts.max() / counts.sum())

        return sum(label_purity) / self.args.num_classes

    def get_pseudo_labels(
        self, embeddings, all_labels, num_clu, outliers_fraction=0.3
        ):  # outliers_fraction = 0.3  # 离群因子
        """
        非群点伪标签算法
        """
        if len(all_labels.shape) != 1:
            all_labels = torch.argmax(all_labels, dim=1)

        P = random.uniform(0.75, 0.95)
        avgclusternum = all_labels.shape[0] // num_clu

        wo_outlier_labels = all_labels.clone()

        # 将每个簇类的标签放在一个列表中
        clu_idx = [[] for i in range(num_clu)]
        for i, la in enumerate(wo_outlier_labels):
            clu_idx[la].append(i)

        outlier_idx = []  # 用于存放 离群索引数据 的 索引
        for idx in clu_idx:
            if len(idx) <= 0: continue
            clf = IsolationForest(contamination=outliers_fraction)
            # clf = LocalOutlierFactor()
            clf.fit(embeddings[idx, :])
            res = clf.predict(embeddings[idx, :])
            outlier_idx += [idx[i] for i, f in enumerate(res) if f != 1]

        #     idx_without_outlierP = [[] for i in range(num_clu)]
        #     for i, la in enumerate(wo_outlier_labels):
        #         if i not in outlier_idx:
        #             idx_without_outlierP[la].append(i)

        #     for idx in idx_without_outlierP:
        #         num_threshold = int(avgclusternum * P)
        #         if len(idx) <= num_threshold: continue
        #         temp = np.random.choice(idx, len(idx) - num_threshold, replace=False)
        #         outlier_idx += temp.tolist()

        wo_outlier_labels[outlier_idx] = -1

        print((wo_outlier_labels != -1).sum(), wo_outlier_labels.shape[0])

        return wo_outlier_labels

    def get_pseudo_labels2(self, embeddings, all_labels, num_clu):
        if len(all_labels.shape) != 1:
            all_labels = torch.argmax(all_labels, dim=1).cpu().detach().numpy()

    def plot_p(self, embeddings, y_pred, step, num_clu=20):
        from sklearn.manifold import TSNE
        tsne = TSNE(learning_rate='auto', init='random')
        # X = np.vstack((embeddings.cpu().detach().numpy(), center.cpu().detach().numpy()))
        X = embeddings
        X_d = tsne.fit_transform(X)

        # way1
        plt.cla()
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(X_d[:, 0], X_d[:, 1], c=y_pred, cmap='nipy_spectral')
        # plt.colorbar(scatter)

        # Hide the axes
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        plt.savefig(f'./pic/{step}_1.png', dpi=1200)
        plt.savefig(f'./pic/{step}_1.pdf', dpi=1200)

        # # 创建散点图
        # plt.cla()
        # plt.figure(figsize=(8, 6))
        # fig, ax = plt.subplots()
        # scatter = ax.scatter(X_d[:,0], X_d[:,1], c=y_pred, s=2, alpha=0.95)
        # # 隐藏坐标轴
        # ax.axis('off')
        # plt.savefig(f'./pic/{step}_2.png', dpi=1200)
        # plt.savefig(f'./pic/{step}_2.pdf', dpi=1200)

        # plt.cla()

    def plot_(self, embeddings, y_pred0, step, num_clu=20):

        y_o = self.get_pseudo_labels(embeddings, y_pred0, num_clu, outliers_fraction=0.1)

        idx = y_o != -1
        y_pred = y_pred0.clone()
        embeddings = embeddings[idx, :]
        y_pred = y_pred[idx]

        from sklearn.manifold import TSNE
        tsne = TSNE(learning_rate='auto', init='random')
        X = embeddings
        X_d = tsne.fit_transform(X)
        plt.figure(figsize=(8, 8))
        # plt.subplot(1, 2, 1)
        plt.scatter(X_d[:, 0], X_d[:, 1], marker='o', c=y_pred, cmap='Spectral', alpha=0.98, s=5)
        # plt.subplot(1, 2, 2)
        plt.axis('off')
        # plt.gca().axes.get_xaxis().set_visible(False)
        # plt.gca().axes.get_yaxis().set_visible(False)
        plt.savefig(f'./pic/{step}_1.png', dpi=1200)
        plt.savefig(f'./pic/{step}_1.pdf', dpi=1200)

        # plt.savefig(self.args.pic_dir+'/'+self.args.data_name+str(epoch)+'-1.pdf')
        # plt.savefig(self.args.pic_dir+'/'+self.args.data_name+str(epoch)+'-1.png')
        # plt.cla()
        # if self.args.class_num <= 10:
        #     sns.scatterplot(x=X_d[:, 0], y=X_d[:, 1], hue=labels_vector, palette="Set2")
        # else:
        #     plt.scatter(X_d[:, 0], X_d[:, 1], c=labels_vector, marker='o', cmap='Spectral', alpha=0.98, s=20)
        # plt.axis('off')
        # plt.savefig(self.args.pic_dir+'/'+self.args.data_name+str(epoch)+'-2.pdf')
        # plt.savefig(self.args.pic_dir+'/'+self.args.data_name+str(epoch)+'-2.png')
