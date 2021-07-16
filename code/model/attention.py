import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from .fc import FCNet, BCNet
import torch.nn.functional as F

class BaseAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(BaseAttention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class UpDnAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(UpDnAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)  # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class SanAttention(nn.Module):
  def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
    super(SanAttention, self).__init__()
    self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
    self.q_lin = nn.Linear(q_features, mid_features)
    self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

    self.drop = nn.Dropout(drop)
    self.relu = nn.LeakyReLU(inplace=True)

  def forward(self, v, q):
    v = self.v_conv(self.drop(v))
    q = self.q_lin(self.drop(q))
    q = tile_2d_over_nd(q, v)
    x = self.relu(v + q)
    x = self.x_conv(self.drop(x))
    return x

def tile_2d_over_nd(feature_vector, feature_map):
  """ Repeat the same feature vector over all spatial positions of a given feature map.
    The feature vector should have the same batch size and number of features as the feature map.
  """
  n, c = feature_vector.size()
  spatial_size = feature_map.dim() - 2
  tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
  return tiled

def apply_attention(input, attention):
  """ Apply any number of attention maps over the input.
    The attention map has to have the same size in all dimensions except dim=1.
  """
  # import pdb
  # pdb.set_trace()
  n, c = input.size()[:2]
  glimpses = attention.size(1)

  # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
  input = input.view(n, c, -1)
  attention = attention.view(n, glimpses, -1)
  s = input.size(2)

  # apply a softmax to each attention map separately
  # since softmax only takes 2d inputs, we have to collapse the first two dimensions together
  # so that each glimpse is normalized separately
  attention = attention.view(n * glimpses, -1)
  attention = F.softmax(attention)

  # apply the weighting by creating a new dim to tile both tensors over
  target_size = [n, glimpses, c, s]
  input = input.view(n, 1, c, s).expand(*target_size)
  attention = attention.view(n, glimpses, 1, s).expand(*target_size)
  weighted = input * attention
  # sum over only the spatial dimension
  weighted_mean = weighted.sum(dim=3)
  # the shape at this point is (n, glimpses, c, 1)
  return weighted_mean.view(n, -1)


class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2, .5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
                                  name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits
