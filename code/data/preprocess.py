import os
import os.path as osp
import re
import random
import itertools
import h5py
import torch
import torch.utils.data as data
import pdb
from torch.utils.data.dataloader import default_collate
from collections import Counter
from PIL import Image
# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))


def invert_dict(d): return {v: k for k, v in d.items()}


def process_punctuation(s):
    # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
    # this version should be faster since we use re instead of repeated operations on str's
    original_s = s
    if _punctuation.search(s) is None:
        return s
    s = _punctuation_with_a_space.sub('', s)
    if re.search(_comma_strip, s) is not None:
        s = s.replace(',', '')
    s = _punctuation.sub(' ', s)
    s = _period_strip.sub('', s)
    if s.strip() == '':
        return original_s.strip()
    else:
        return s.strip()


def extract_vocab(iterable, top_k=None, start=0, input_vocab=None):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)

    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


class CocoImages(data.Dataset):
    def __init__(self, path, transform=None):
        super(CocoImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)


class Composite(data.Dataset):
    """ Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset. """

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        current = self.datasets[0]
        for d in self.datasets:
            if item < len(d):
                return d[item]
            item -= len(d)
        else:
            raise IndexError('Index too large for composite dataset')

    def __len__(self):
        return sum(map(len, self.datasets))

    def _get_answer_vectors(self, answer_indices):
        return self.datasets[0]._get_answer_vectors(answer_indices)

    def _get_answer_sequences(self, answer_indices):
        return self.datasets[0]._get_answer_sequences(answer_indices)

    @property
    def vector(self):
        return self.datasets[0].vector

    @property
    def token_to_index(self):
        return self.datasets[0].token_to_index

    @property
    def answer_to_index(self):
        return self.datasets[0].answer_to_index

    @property
    def index_to_answer(self):
        return self.datasets[0].index_to_answer

    @property
    def num_tokens(self):
        return self.datasets[0].num_tokens

    @property
    def num_answer_tokens(self):
        return self.datasets[0].num_answer_tokens

    @property
    def vocab(self):
        return self.datasets[0].vocab


def eval_collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)
