# The following code is modified based on https://github.com/pytorch/text/blob/master/torchtext/vocab.py
import array
import zipfile
from tqdm import tqdm
from six.moves.urllib.request import urlretrieve
import os
import os.path as osp
import torch
import io
# from ansemb.config import data_root, cfg


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner


class Vector(object):
    def __init__(self, cache_path,
                 vector_type='glove.840B', unk_init=torch.Tensor.zero_) -> object:
        urls = {
            'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
            'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
            'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }
        url = urls[vector_type] if urls.get(vector_type, False) != False else None
        name = osp.splitext(osp.basename(url))[0] + '.txt'  # glove.840B.300d.txt

        self.unk_init = unk_init
        self.cache(name, cache_path, url=url)

    def __getitem__(self, token):
        if self.stoi.get(token, -1) != -1:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(1, self.dim))

    def _prepare(self, vocab):
        word2vec = torch.Tensor(len(vocab), self.dim)
        for token, idx in vocab.items():
            word2vec[idx, :] = self[token]

        return word2vec

    def check(self, token):
        if self.stoi.get(token, -1) != -1:
            return True
        else:
            return False

    def cache(self, name, cache_path, url=None):
        # cache_path='.vector_cache',
        #name= "glove.840B.300d.txt"
        #url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'

        path = osp.join(cache_path, name)
        path_pt = "{}.pt".format(path)

        if not osp.isfile(path_pt):
            # download vocab file if it does not exists
            if not osp.exists(path) and url:
                dest = osp.join(cache_path, os.path.basename(url))
                if not osp.exists(dest):  # 连这个压缩包都不存在
                    print('[-] Downloading vectors from {}'.format(url))  # 下载
                    if not osp.exists(cache_path):
                        os.mkdir(cache_path)  # 新建这个cache文件夹

                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=dest) as t:
                        urlretrieve(url, dest, reporthook=reporthook(t))  # 下载

                print('[-] Extracting vectors into {}'.format(path))  # 提取vector到txt中
                ext = os.path.splitext(dest)[1][1:]
                if ext == 'zip':
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(cache_path)  # 解压

            if not os.path.isfile(path):
                raise RuntimeError('no vectors found at {}'.format(path))

            # build vocab list
            itos, vectors, dim = [], array.array(str('d')), None

            # Try to read the whole file with utf-8 encoding.
            binary_lines = False
            try:
                with io.open(path, encoding="utf8") as f:
                    lines = [line for line in f]
            # If there are malformed lines, read in binary mode
            # and manually decode each word from utf-8
            except:
                print("[!] Could not read {} as UTF8 file, "
                      "reading file as bytes and skipping "
                      "words with malformed UTF8.".format(path))
                with open(path, 'rb') as f:
                    lines = [line for line in f]
                binary_lines = True

            print("[-] Loading vectors from {}".format(path))  # 读取vector
            for line in tqdm(lines, total=len(lines)):
                # Explicitly splitting on " " is important, so we don't
                # get rid of Unicode non-breaking spaces in the vectors.
                entries = line.rstrip().split(" ")
                word, entries = entries[0], entries[1:]
                if dim is None and len(entries) > 1:
                    dim = len(entries)  # dim 由向量长度决定
                elif len(entries) == 1:
                    print("Skipping token {} with 1-dimensional "
                          "vector {}; likely a header".format(word, entries))
                    continue
                elif dim != len(entries):  # 向量长度不等
                    raise RuntimeError(
                        "Vector for token {} has {} dimensions, but previously "
                        "read vectors have {} dimensions. All vectors must have "
                        "the same number of dimensions.".format(word, len(entries), dim))

                vectors.extend(float(x) for x in entries)  # 向量转float，存到vector中
                itos.append(word)  # 词也加入 index to s

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}  # s to index
            self.vectors = torch.Tensor(vectors).view(-1, dim)  # 转tensor
            self.dim = dim
            print('* Caching vectors to {}'.format(path_pt))  # 存到 pt文件中
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:  # 如果py文件存在，就直接读出
            print('* Loading vectors to {}'.format(path_pt))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)
