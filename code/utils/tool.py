import os
import json

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pdb


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


def unseen_mask(args, val_loader):
    negtive_mux = None
    # zsl
    if args.ZSL == 1:
        negtive_mux = torch.ones(args.TEST.batch_size, args.FVQA.max_ans)
        indices = val_loader.dataset.answer_indices
        all_ans = set(aid for aids in indices for aid in aids)

        # unseen 类置0
        for i in all_ans:
            for j in range(args.TRAIN.batch_size):
                negtive_mux[j, i] = 0
        negtive_mux = negtive_mux * (-1e13)
        negtive_mux = negtive_mux.cuda()
        # pdb.set_trace()
    return negtive_mux


def cosine_sim(im, s):
    return im.mm(s.t())


def batch_mc_acc(predicted):
    """ Compute the accuracies for a batch of predictions and answers """
    N, C = predicted.squeeze().size()
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    return (predicted_index == C - 1).float()


def batch_top1(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    return true.gather(dim=1, index=predicted_index).clamp(max=1)


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    # import pdb
    # pdb.set_trace()
    # _, predicted_index = predicted.max(dim=1, keepdim=True)
    # agreeing = true[0].gather(dim=1, index=predicted_index)
    # return (agreeing * 0.3).clamp(max=1)
    if len(true.shape) == 3:
        true = true[0]
    _, ok = predicted.topk(10, dim=1)
    agreeing_all = torch.zeros([predicted.shape[0], 1], dtype=torch.float).cuda()
    for i in range(10):
        tmp = ok[:, i].reshape(-1, 1)
        agreeing_all += true.gather(dim=1, index=tmp)
        if i == 0:
            top1 = (agreeing_all * 0.3).clamp(max=1)
        if i == 2:
            top3 = (agreeing_all * 0.3).clamp(max=1)
        if i == 9:
            top10 = (agreeing_all * 0.3).clamp(max=1)

    top1 = top1.sum().item() / top1.shape[0]
    top3 = top3.sum().item() / top3.shape[0]
    top10 = top10.sum().item() / top10.shape[0]
    return top1, top3, top10


# def update_learning_rate(optimizer, epoch):
#     learning_rate = cfg.TRAIN.base_lr * 0.5 ** (float(epoch) / cfg.TRAIN.lr_decay)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = learning_rate
#
#     return learning_rate


class Tracker:
    """ Keep track of results over time, while having access to monitors to display information about them. """

    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}

    class ListStorage:
        """ Storage of data points that updates the given monitors """

        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_features, self.next_targets, _ = next(self.loader)
        except StopIteration:
            self.next_features = None
            self.next_targets = None
            return
        # self.next_features_gpu = []
        # self.next_targets_gpu = {}
        # for xaf in self.next_features:
        #     self.next_features_gpu.append(torch.empty_like(xaf, device='cuda'))
        # for key in self.next_targets.keys():
        #     self.next_targets_gpu[key] = torch.empty_like(self.next_targets[key], device='cuda')
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_features = [single_feature.cuda(non_blocking=True) for single_feature in self.next_features]
            if isinstance(self.next_targets, dict):
                for key in self.next_targets.keys():
                    self.next_targets[key] = self.next_targets[key].cuda(non_blocking=True)
            else:
                self.next_targets = [single_target.cuda(non_blocking=True) for single_target in self.next_targets]
            # for index in range(len(self.next_features_gpu)):
            #     self.next_features_gpu[index].copy_(self.next_features[index], non_blocking=True)
            # for key in self.next_targets_gpu.keys():
            #     self.next_targets_gpu[key].copy_(self.next_targets[key], non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        # features = self.next_features_gpu
        # targets = self.next_targets_gpu
        features = self.next_features
        targets = self.next_targets
        if features is not None:
            features = [xaf.record_stream(torch.cuda.current_stream()) for xaf in features]
        if targets is not None:
            targets = [targets[xaf].record_stream(torch.cuda.current_stream()) for xaf in targets.keys()]
        self.preload()
        return features, targets


def get_transform(target_size, central_fraction=1.0):
    return transforms.Compose([
        transforms.Scale(int(target_size / central_fraction)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def dele_a(answer):  # 去冠词
    answer_t = answer.replace('.', '')
    answer_tt = answer_t.replace(',', '')
    answer_ttt = answer_tt.replace("the ", "")
    answer_tttt = answer_ttt.replace("an ", "")
    answer_ttttt = answer_tttt.replace("a ", "")
    ans_list = [answer_t, answer_tt, answer_ttt, answer_tttt, answer_ttttt]

    return list(set(ans_list))


def transfer(answer):  # 单复数转换
    tokens = word_tokenize(answer)
    tagged_sent = pos_tag(tokens)
    wnl = WordNetLemmatizer()

    new = []
    for tag in tagged_sent:
        if tag[0] == "as":
            new.append("as")
            continue
        elif tag[0] == "grazing" or tag[0] == "timing" or tag[0] == "bicycling":
            kk = tag[0].replace("ing", "") + "e"
            new.append(kk)
            continue
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        tmp = wnl.lemmatize(tag[0], pos=wordnet_pos)
        if tmp == "ax":
            tmp = "axe"
        elif tmp == "people":
            tmp = "person"
        elif tmp == "teeth":
            tmp = "tooth"
        elif tmp == "worn":
            tmp = "wear"
        new.append(tmp)  # 词形还原
    string = ' '
    key = string.join(new)
    return key


def hand_remove(answer):  # 手动去ing，s，es
    _ing = answer.replace("ing", "")
    __ing = answer.replace("ing ", " ")
    _s = answer.replace("s", "")
    __s = answer.replace("s ", " ")
    _es = answer.replace("es", "")
    __es = answer.replace("es ", " ")
    _er = answer.replace("er", "")
    __er = answer.replace("er ", " ")
    return list(set([_ing, _s, _es, _er, __ing, __s, __es, __er]))


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def deal_fact(dic, fact):
    fact = fact.split('/')
    if fact[-1] == "n" or fact[-1] == "v":
        ans = fact[-2]
    else:
        ans = fact[-1]

    ans = ans.split(':')
    if ans[0] == "Category":
        ans = ans[1]
    else:
        ans = ans[0]

    # if ans[-1] == ")":
    #     # ans = ans.split("(")[0]
    #     pdb.set_trace()
    #     ans = dic["answer"]
    return ans
