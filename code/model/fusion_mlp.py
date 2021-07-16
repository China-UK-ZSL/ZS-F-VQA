import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils import freeze_layer
from torch.autograd import Variable
from .fc import GroupMLP
from .language_model import WordEmbedding


class MLP(nn.Module):
    #args, self.train_loader.dataset, self.question_word2vec
    # def __init__(self, args, dataset, question_word2vec):
    def __init__(self, args, dataset, embedding_weights=None, rnn_bidirectional=True):
        super(MLP, self).__init__()
        embedding_requires_grad = not args.freeze_w2v  # freeze 则不需要grad
        question_features = 300
        vision_features = args.output_features  # 图片的

        # self.text = BagOfWordsMLPProcessor(
        self.text = BagOfWordsProcessor(
            embedding_tokens=embedding_weights.size(0) if embedding_weights is not None else dataset.num_tokens,
            embedding_weights=embedding_weights,
            embedding_features=300,
            embedding_requires_grad=embedding_requires_grad,
        )
        self.mlp = GroupMLP(
            in_features=vision_features + question_features,
            mid_features= 4 * args.hidden_size,
            out_features=args.embedding_size,
            drop=0.5,
            groups=64,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, b, q, q_len):
        q = F.normalize(self.text(q, list(q_len.data)), p=2, dim=1)  # 问题向量求平均值
        v = F.normalize(F.avg_pool2d(v, (v.size(2), v.size(3))).squeeze(), p=2, dim=1)

        combined = torch.cat([v, q], dim=1)
        embedding = self.mlp(combined)
        return embedding


class BagOfWordsProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features,
                 embedding_weights, embedding_requires_grad):
        super(BagOfWordsProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.embedding.weight.data = embedding_weights
        self.embedding.weight.requires_grad = embedding_requires_grad

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        q_len = Variable(torch.Tensor(q_len).view(-1, 1) + 1e-12, requires_grad=False).cuda()

        return torch.div(torch.sum(embedded, 1), q_len)
