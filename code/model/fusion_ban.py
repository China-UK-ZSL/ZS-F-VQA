"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is adapted from: https://github.com/jnhwkim/ban-vqa (written by Jin-Hwa Kim)
"""
import torch.nn as nn

from .attention import BiAttention
from .classifier import SimpleClassifier
from .counting import Counter
from .fc import FCNet, BCNet
from .language_model import WordEmbedding, UpDnQuestionEmbedding
from utils import freeze_layer


class BAN(nn.Module):
    #args, self.train_loader.dataset, self.question_word2vec
    # def __init__(self, args, dataset, question_word2vec):
    def __init__(self, args, dataset, question_word2vec):
        super(BAN, self).__init__()
        self.args = args
        self.w_emb = WordEmbedding(question_word2vec.size(0), 300, .0)
        if args.freeze_w2v:
            self.w_emb.init_embedding(question_word2vec)
            freeze_layer(self.w_emb)
        self.q_emb = UpDnQuestionEmbedding(300, args.embedding_size, 1, False, .0)
        self.v_att = BiAttention(args.v_dim, self.q_emb.num_hid, self.q_emb.num_hid, args.glimpse)
        self.b_net = []
        self.q_prj = []
        self.c_prj = []
        self.objects = 10  # minimum number of boxes
        for i in range(args.glimpse):
            self.b_net.append(BCNet(args.v_dim, self.q_emb.num_hid, self.q_emb.num_hid, None, k=1))
            self.q_prj.append(FCNet([self.q_emb.num_hid, self.q_emb.num_hid], '', .2))
            self.c_prj.append(FCNet([self.objects + 1, self.q_emb.num_hid], 'ReLU', .0))

        self.b_net = nn.ModuleList(self.b_net)
        self.q_prj = nn.ModuleList(self.q_prj)
        self.c_prj = nn.ModuleList(self.c_prj)
        self.counter = Counter(self.objects)
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, b, q, q_len):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        boxes = b[:, :, :4].transpose(1, 2)

        b_emb = [0] * self.args.glimpse
        att, logits = self.v_att.forward_all(v, q_emb)  # b x g x v x q

        for g in range(self.args.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:, g, :, :])  # b x l x h

            atten, _ = logits[:, g, :, :].max(2)
            embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        return q_emb.sum(1)
