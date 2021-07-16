from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = d_.transpose(1, 2).transpose(2, 3)  # b x v x q x h_dim
            return logits

        # broadcast Hadamard product, matrix-matrix production
        # fast computation but memory inefficient
        # epoch 1, time: 157.84
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat  # broadcast, b x h_out x v x h_dim
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2, 3))  # b x h_out x v x q
            logits = logits + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        # epoch 1, time: 304.87
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1, 2).unsqueeze(2)  # b x d x 1 x v
        q_ = self.q_net(q).transpose(1, 2).unsqueeze(3)  # b x d x q x 1
        logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)  # b x d x 1 x 1
        logits = logits.squeeze(3).squeeze(2)
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits


class GroupMLP(nn.Module):
    def __init__(self, in_features, mid_features, out_features, drop=0.5, groups=1):
        super(GroupMLP, self).__init__()

        self.conv1 = nn.Conv1d(in_features, mid_features, 1)
        self.drop = nn.Dropout(p=drop)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(mid_features, out_features, 1, groups=groups)

    def forward(self, a):
        N, C = a.size()
        h = self.relu(self.conv1(a.view(N, C, 1)))
        return self.conv2(self.drop(h)).view(N, -1)


class GroupMLP_1lay(nn.Module):
    def __init__(self, in_features, mid_features, out_features, drop=0.5, groups=1):
        super(GroupMLP_1lay, self).__init__()

        self.conv1 = nn.Conv1d(in_features, mid_features, 1)
        self.batch_norm_fusion = nn.BatchNorm1d(mid_features, affine=False)
        self.drop = nn.Dropout(p=drop)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(mid_features, out_features, 1, groups=groups)

    def forward(self, a):
        N, C = a.size()
        h = self.conv1(a.view(N, C, 1))
        h = self.batch_norm_fusion(h)
        h = self.relu(h)
        return self.conv2(self.drop(h)).view(N, -1)


class GroupMLP_2lay(nn.Module):
    def __init__(self, in_features, mid_features, out_features, drop=0.5, groups=1):
        super(GroupMLP_2lay, self).__init__()

        self.conv1 = nn.Conv1d(in_features, mid_features, 1)
        self.batch_norm_fusion = nn.BatchNorm1d(mid_features, affine=False)
        self.drop = nn.Dropout(p=drop)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(mid_features, mid_features, 1, groups=groups)
        self.conv3 = nn.Conv1d(mid_features, out_features, 1, groups=groups)

    def forward(self, a):
        N, C = a.size()
        h = self.conv1(a.view(N, C, 1))
        h = self.relu(h)
        h = self.conv2(h)
        h = self.batch_norm_fusion(h)
        h = self.relu(h)
        return self.conv3(self.drop(h)).view(N, -1)
