import torch
import torch.nn as nn
from .language_model import WordEmbedding, UpDnQuestionEmbedding
from .attention import UpDnAttention
from .classifier import SimpleClassifier
from .fc import FCNet
from utils import freeze_layer

class UD(nn.Module):
    def __init__(self, args, dataset, question_word2vec):
        super(UD, self).__init__()
        self.w_emb = WordEmbedding(question_word2vec.size(0), 300, 0.0)
        if args.freeze_w2v:
            self.w_emb.init_embedding(question_word2vec)
            freeze_layer(self.w_emb)
            # self.w_emb.weight.requires_grad = False

        self.q_emb = UpDnQuestionEmbedding(300, args.embedding_size, 1, False, 0.0)
        self.v_att = UpDnAttention(args.v_dim, self.q_emb.num_hid, args.embedding_size)
        self.q_net = FCNet([self.q_emb.num_hid, args.embedding_size])
        self.v_net = FCNet([args.v_dim, args.embedding_size])
        # self.classifier = SimpleClassifier(
        #     args.embedding_size, args.embedding_size * 2, args.num_ans_candidates, 0.5)

    def forward(self, v, b, q, qlen):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # print("q = {}".format(q))
        w_emb = self.w_emb(q)
        # print("w_emb = {}".format(w_emb))
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb) # [spa, 1]
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
       # logits = self.classifier(joint_repr)
        return joint_repr
