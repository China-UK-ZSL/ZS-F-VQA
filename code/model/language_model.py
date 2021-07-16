import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.init as init
import pdb


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """

    def __init__(self, ntoken, emb_dim, dropout=0):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        # weight_init = torch.from_numpy(np.load(np_file))
        weight_init = np_file
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class UpDnQuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout=0, rnn_type='GRU'):
        """Module for question embedding
        """
        super(UpDnQuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers=1, bidirect=True, dropout=0, rnn_type='GRU', words_dropout=None,
                 dropout_before_rnn=None,
                 dropout_after_rnn=None):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.bidirect = bidirect
        self.ndirections = 1 + int(bidirect)
        if bidirect:
            num_hid = int(num_hid / 2)
        self.words_dropout = words_dropout
        if dropout_before_rnn is not None:
            self.dropout_before_rnn = nn.Dropout(p=dropout_before_rnn)
        else:
            self.dropout_before_rnn = None
        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)
        if dropout_after_rnn is not None:
            self.dropout_after_rnn = nn.Dropout(p=dropout_after_rnn)
        else:
            self.dropout_after_rnn = None

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x, qlen=None):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        num_tokens = x.size(1)
        if self.words_dropout is not None and self.words_dropout > 0:
            num_dropout = int(self.words_dropout * num_tokens)
            rand_ixs = np.random.randint(0, num_tokens, (batch, num_dropout))
            for bix, token_ixs in enumerate(rand_ixs):
                x[bix, token_ixs] *= 0
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        if self.dropout_before_rnn is not None:
            x = self.dropout_before_rnn(x)

        q_words_emb, hidden = self.rnn(x, hidden)  # q_words_emb: B x num_words x gru_dim, hidden: 1 x B x gru_dim

        out = None
        if self.bidirect:
            forward_ = q_words_emb[:, -1, :self.num_hid]
            backward = q_words_emb[:, 0, self.num_hid:]
            hid = torch.cat((forward_, backward), dim=1)
            out = hid
            # return q_words_emb, hid
        else:
            out = q_words_emb[:, -1]
            # return q_words_emb, q_words_emb[:, -1]

        if self.dropout_after_rnn is not None:
            out = self.dropout_after_rnn(out)
        return out

class Seq2SeqRNN(nn.Module):
  def __init__(self, input_features, rnn_features, num_layers=1, drop=0.0,
               rnn_type='LSTM', rnn_bidirectional=False):
    super(Seq2SeqRNN, self).__init__()
    self.bidirectional = rnn_bidirectional

    if rnn_type == 'LSTM':
      self.rnn = nn.LSTM(input_size=input_features,
                hidden_size=rnn_features, dropout=drop,
                num_layers=num_layers, batch_first=True,
                bidirectional=rnn_bidirectional)
    elif rnn_type == 'GRU':
      self.rnn = nn.GRU(input_size=input_features,
                hidden_size=rnn_features, dropout=drop,
                num_layers=num_layers, batch_first=True,
                bidirectional=rnn_bidirectional)
    else:
      raise ValueError('Unsupported Type')

    self.init_weight(rnn_bidirectional, rnn_type)

  def init_weight(self, bidirectional, rnn_type):
    self._init_rnn(self.rnn.weight_ih_l0, rnn_type)
    self._init_rnn(self.rnn.weight_hh_l0, rnn_type)
    self.rnn.bias_ih_l0.data.zero_()
    self.rnn.bias_hh_l0.data.zero_()

    if bidirectional:
      self._init_rnn(self.rnn.weight_ih_l0_reverse, rnn_type)
      self._init_rnn(self.rnn.weight_hh_l0_reverse, rnn_type)
      self.rnn.bias_ih_l0_reverse.data.zero_()
      self.rnn.bias_hh_l0_reverse.data.zero_()

  def _init_rnn(self, weight, rnn_type):
    chunk_size = 4 if rnn_type == 'LSTM' else 3
    for w in weight.chunk(chunk_size, 0):
      init.xavier_uniform(w)

  def forward(self, q_emb, q_len):
    lengths = torch.LongTensor(q_len)
    lens, indices = torch.sort(lengths, 0, True)

    packed = pack_padded_sequence(q_emb[indices.cuda()], lens.tolist(), batch_first=True)
    if isinstance(self.rnn, nn.LSTM):
        # pdb.set_trace()
        _, ( outputs, _ ) = self.rnn(packed)
    elif isinstance(self.rnn, nn.GRU):
        _, outputs = self.rnn(packed)

    if self.bidirectional:
      outputs = torch.cat([ outputs[0, :, :], outputs[1, :, :] ], dim=1)
    else:
      outputs = outputs.squeeze(0)

    _, _indices = torch.sort(indices, 0)
    outputs = outputs[_indices.cuda()]

    return outputs