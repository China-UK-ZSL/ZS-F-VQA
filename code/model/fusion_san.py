import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils import freeze_layer
from torch.autograd import Variable
from .attention import SanAttention, apply_attention
from .fc import GroupMLP
from .language_model import Seq2SeqRNN, WordEmbedding
import pdb

class SAN(nn.Module):
    #args, self.train_loader.dataset, self.question_word2vec
    #def __init__(self, args, dataset, question_word2vec):
    def __init__(self, args, dataset,embedding_weights=None,rnn_bidirectional=True):
        super(SAN, self).__init__()
        embedding_requires_grad = not args.freeze_w2v
        question_features = 1024
        rnn_features = int(question_features // 2) if rnn_bidirectional else int(question_features)
        vision_features = args.output_features
        glimpses = 2

        # vocab_size = embedding_weights.size(0)
        # vector_dim = embedding_weights.size(1)
        # self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)
        # self.embedding.weight.data = embedding_weights
        # self.embedding.weight.requires_grad = embedding_requires_grad
        self.w_emb = WordEmbedding(embedding_weights.size(0), 300, .0)
        if args.freeze_w2v:
            self.w_emb.init_embedding(embedding_weights)
            freeze_layer(self.w_emb)

        self.drop = nn.Dropout(0.5)
        self.text = Seq2SeqRNN(
            input_features=embedding_weights.size(1),
            rnn_features=int(rnn_features),
            rnn_type='LSTM',
            rnn_bidirectional=rnn_bidirectional,
        )
        self.attention = SanAttention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )
        self.mlp = GroupMLP(
            in_features=glimpses * vision_features + question_features,
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
        # pdb.set_trace()
        q = self.text(self.drop(self.w_emb(q)), list(q_len.data))
        # q = self.text(self.embedding(q), list(q_len.data))

        v = F.normalize(v, p=2, dim=1)
        a = self.attention(v, q)
        v = apply_attention(v, a)

        combined = torch.cat([v, q], dim=1)
        embedding = self.mlp(combined)
        return embedding