import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pdb
import model.fc as FC
from .fc import GroupMLP, GroupMLP_2lay, GroupMLP_1lay


class MLP(nn.Module):
    def __init__(self, args, dataset):
        super(MLP, self).__init__()
        ans_net_list = ["GroupMLP", "GroupMLP_1lay", "GroupMLP_2lay"]
        ans_net = ans_net_list[args.ans_net_lay]
        self.mlp = getattr(FC, ans_net)(
            in_features=args.ans_feature_len,  # fan
            mid_features=args.hidden_size,  # 2048
            out_features=args.embedding_size,  # fan
            drop=0.0,
            groups=64,  # 64
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, a, a_len=None):
        # pdb.set_trace()
        return self.mlp(F.normalize(a, p=2))
