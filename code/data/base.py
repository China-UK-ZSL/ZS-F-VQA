import json
import os
import os.path as osp
import nltk
import h5py
import torch
import torch.utils.data as data
import pdb
from nltk import word_tokenize, pos_tag
import re
import numpy as np
import sys
import pickle as pkl

################
from .preprocess import invert_dict


class VisualQA(data.Dataset):
    def __init__(self,
                 args,
                 vector):
        super(VisualQA, self).__init__()

        # vocab
        self.vector = vector
        self.args = args
        # process question
        # self.args.question_vocab_path = osp.join(project_root, 'data', 'question.vocab.json') # a joint question vocab across all dataset
        with open(self.args.question_vocab_path, 'r') as fd:
            question_vocab = json.load(fd)
        self.token_to_index = question_vocab['question']
        self._max_question_length = question_vocab['max_question_length']
        self.image_features_path = args.FVQA.feature_path
        self.index_to_token = invert_dict(self.token_to_index)

        answer_vocab_path = self.args.FVQA.answer_vocab_path
        fact_vocab_path = self.args.FVQA.fact_vocab_path
        relation_vocab_path = self.args.FVQA.relation_vocab_path

        if self.args.fact_map:
            with open(fact_vocab_path, 'r') as fd:
                answer_vocab = json.load(fd)
        elif self.args.relation_map:
            with open(relation_vocab_path, 'r') as fd:
                answer_vocab = json.load(fd)
        else:
            with open(answer_vocab_path, 'r') as fd:
                answer_vocab = json.load(fd)
        self.answer_to_index = answer_vocab['answer']
        self.index_to_answer = invert_dict(self.answer_to_index)

        self.cached_answers_g2v = {}  # 只编码KGE
        self.cached_answers_w2v = {}  # 只编码序列
        self.cached_answers_gae = {}
        self.cached_answers_bert = {}
        self.unk_vector = self.vector['UNK']
        if "KG" in self.args.method_choice:
            self._map_kg()
        if "GAE" in self.args.method_choice:
            # self._map_gae()
            self._map_bert()

    @property
    def max_question_length(self):
        return self._max_question_length

    @property
    def max_answer_length(self):
        assert hasattr(self, answers), 'Dataloader must have access to answers'
        if not hasattr(self, '_max_answer_length'):
            self._max_answer_length = max(map(len, self.answers))
        return self._max_answer_length

    @property
    def num_tokens(self):
        return len(self.token_to_index)

    @property
    def num_answers(self):
        return len(self.answer_to_index)

    def __len__(self):
        return len(self.questions)

    # Internal data utility---------------------------------------

    def _load_image(self, image_id):
        """ Load an image """
        # pdb.set_trace()
        index = self.image_id_to_index[image_id]
        spa = torch.zeros([1, 1])  # init

        if self.args.fusion_model == 'UD' or self.args.fusion_model == 'BAN':
            spatials = self.features_file['spatial_features']
            dataset = self.features_file['image_features']  # 直接读取特征文件
            spa = spatials[index].astype('float32')
            spa = torch.from_numpy(spa)
        else:
            dataset = self.features_file['features']  # 直接读取特征文件

        img = dataset[index].astype('float32')

        return torch.from_numpy(img), spa

    def _create_image_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path, 'r')

        if self.args.fusion_model == 'UD' or self.args.fusion_model == 'BAN':
            import _pickle as cPickle
            image_id_to_index = cPickle.load(open(self.args.FVQA.img_id2idx, "rb"))
            # pdb.set_trace()
            self.s_dim = self.features_file['spatial_features'].shape[2]
            self.v_dim = self.features_file['image_features'].shape[2]  # 直接读取特征文件

        else:
            with h5py.File(self.image_features_path, 'r') as features_file:
                image_ids = features_file['ids'][()]
            image_id_to_index = {id: i for i, id in enumerate(image_ids)}
        return image_id_to_index

    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.zeros(self.max_question_length).long()
        for i, token in enumerate(question):
            index = self.token_to_index.get(token, 0)
            vec[i] = index
        return vec, len(question)

    def _map_kg(self):
        if "KG" not in self.args.method_choice:
            return
        # print("using kg embedding")
        kg_path = self.args.FVQA.kg_path
        entity_path = self.args.FVQA.entity_path  # 来源中的词对应的向量
        relation_path = self.args.FVQA.relation_path  # 同上
        relation2id_path = self.args.FVQA.relation2id_path  # 搜寻候选答案的来源
        entity2id_path = self.args.FVQA.entity2id_path  # 搜寻候选答案的来源

        a = np.load(entity_path)
        b = np.load(relation_path)
        self.map_kg = np.vstack((a, b))

        # 随机得到一个矩阵，以模拟随机的情况
        # self.map_ran=torch.zeros(self.map_kg.shape)
        # self.map_ran = torch.rand(self.map_kg.shape)
        # self.map_ran = torch.randn(self.map_kg.shape)
        # self.map_kg = self.map_ran

        self.map_kg = torch.Tensor(self.map_kg).view(-1, 300)

        self.stoi_kg = {}
        with open(os.path.join(entity2id_path), "r") as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                line = re.split('\t|\n', line)[:2]
                self.stoi_kg[line[0]] = int(line[1])
        sz = len(self.stoi_kg)
        with open(os.path.join(relation2id_path), "r") as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                line = re.split('\t|\n', line)[:2]
                self.stoi_kg[line[0]] = int(line[1]) + sz

    def _map_gae(self):
        if "GAE" not in self.args.method_choice:
            return
        # print("using kg embedding")

        _gae_path = self.args.FVQA.gae_path
        gae_path = osp.join(_gae_path, str(self.args.FVQA.gae_node_num) + "_init_" + self.args.FVQA.gae_init + ".pkl")
        print("gae file:", gae_path)
        with open(gae_path, 'rb') as f:
            if sys.version_info > (3, 0):
                features = pkl.load(f, encoding='latin1')
            else:
                features = pkl.load(f)
        # 下标到gae向量的映射
        self.map_gae = torch.FloatTensor(np.array(features)).view(-1, 300)
        vertices_f = osp.join(_gae_path, "g_nodes_" + str(self.args.FVQA.gae_node_num) + ".json")
        self.stoi_gae = {}
        with open(vertices_f) as fp:
            vertices_list = json.load(fp)

        for i, vertex in enumerate(vertices_list):
            self.stoi_gae[vertex] = i
        # print("test map gae")
        # pdb.set_trace()

    def _map_bert(self):
        if "GAE" not in self.args.method_choice:
            return
        # print("using kg embedding")

        cache_path = osp.join(self.args.FVQA.bert_path, "map_bert.pt")
        if not osp.exists(cache_path):
            _bert_path = self.args.FVQA.bert_path

            bert_path = osp.join(_bert_path, "conceptnet_bert_embeddings.pt")
            print("bert file:", bert_path)
            _cache = torch.load(bert_path)  # torch.Size([78334, 1024])

            self.map_bert = torch.FloatTensor(self.args.FVQA.max_ans, self.args.bert_dim)
            # 下标到gae向量的映射
            all = []

            with open(osp.join(_bert_path, "cn_node_names_for_embeddings.txt"), 'r', encoding='utf-8') as f:
                while 1:
                    line = f.readline()
                    if not line:
                        break
                    line = re.split('\n', line)
                    all.append(line[0])

            self.stoi_bert = {}  # answer to vector文件的 id 下标
            for key, value in self.answer_to_index.items():
                self.stoi_bert[key] = value
                if key in all:
                    self.map_bert[value] = _cache[all.index(key), :]
                else:
                    cnt = 0.0
                    tmp = torch.zeros(1, self.args.bert_dim).cuda()
                    for i, j in enumerate(all):
                        if len(j) >= 4 and len(key) >= 3 and (key in j or j in key):
                            # pdb.set_trace()
                            tmp += _cache[i, :]  # 取平均
                            cnt += 1
                        if cnt >= 3:
                            break
                    if cnt == 0:
                        raise TypeError('cnt can not = 0 !!!')
                    self.map_bert[value] = tmp / (cnt + 1e-12)

            if (self.map_bert != self.map_bert).any():
                raise TypeError('cnt can not = 0 !!!')
            # pdb.set_trace()
            torch.save({'map_bert': self.map_bert, 'stoi_bert': self.stoi_bert}, cache_path)
        else:
            _cache = torch.load(cache_path)
            self.map_bert = _cache['map_bert']  # 词向量列表 + 长度
            self.stoi_bert = _cache['stoi_bert']  # 答案下标

        # print("test map gae")
        # pdb.set_trace()

    def _get_answer_vectors(self, ways, answer_indices):
        dim = self.vector.dim
        if ways == 'GAE':
            dim = self.args.bert_dim
            return self._encode_answer_vector(self._encode_answer_vector_bert, dim, answer_indices)
            # return self._encode_answer_vector(self._encode_answer_vector_gae, dim, answer_indices)
        elif ways == 'KG':
            return self._encode_answer_vector(self._encode_answer_vector_g2v, dim, answer_indices)
        elif ways == 'W2V':
            return self._encode_answer_vector(self._encode_answer_vector_w2v, dim, answer_indices)

    def _encode_answer_vector(self, encode_model, dim, answer_indices):
        if isinstance(answer_indices[0], list):
            N, C = len(answer_indices), len(answer_indices[0])
            vector = torch.zeros(N, C, dim)
            for i, answer_ids in enumerate(answer_indices):
                for j, answer_id in enumerate(answer_ids):
                    if answer_id != -1:
                        vector[i, j, :] = encode_model(self.index_to_answer[answer_id])
                    else:
                        vector[i, j, :] = self.unk_vector
        else:
            vector = torch.zeros(len(answer_indices), dim)
            for idx, answer_id in enumerate(answer_indices):

                if answer_id != -1:
                    if type(answer_id).__name__ == 'int':
                        vector[idx, :] = encode_model(self.index_to_answer[answer_id])
                    else:
                        vector[idx, :] = encode_model(self.index_to_answer[answer_id.item()])
                else:
                    vector[idx, :] = self.unk_vector
        return vector, []

    def _get_answer_sequences_w2v(self, answer_indices):
        seqs, lengths = [], []
        max_seq_length = 0
        if isinstance(answer_indices[0], list):
            N, C = len(answer_indices), len(answer_indices[0])
            for i, answer_ids in enumerate(answer_indices):
                _seqs = []
                for j, answer_id in enumerate(answer_ids):
                    if answer_id != -1:
                        _seqs.append(self._encode_answer_sequence_w2v(self.index_to_answer[answer_id]))
                    else:
                        _seqs.append([self.unk_vector])
                    if max_seq_length < len(_seqs[-1]):
                        max_seq_length = len(_seqs[-1])  # determing max length
                seqs.append(_seqs)

            vector = torch.zeros(N, C, max_seq_length, self.vector.dim)
            for i, _seqs in enumerate(seqs):
                for j, seq in enumerate(_seqs):
                    if len(seq) != 0:
                        vector[i, j, :len(seq), :] = torch.cat(seq, dim=0)
                    lengths.append(len(seq))
            assert len(lengths) == N * \
                C, 'Wrong lengths - length: {} vs N: {}, C: {} vs seqs: {}'.format(len(lengths), N, C, len(seqs))
        else:
            for idx, answer_id in enumerate(answer_indices):
                if answer_id != -1:
                    if type(answer_id).__name__ == 'int':
                        seqs.append(self._encode_answer_sequence_w2v(self.index_to_answer[answer_id]))
                    else:
                        seqs.append(self._encode_answer_sequence_w2v(self.index_to_answer[answer_id.item()]))
                else:
                    seqs.append([self.unk_vector])

                if max_seq_length < len(seqs[-1]):
                    max_seq_length = len(seqs[-1])  # determing max length

            vector = torch.zeros(len(answer_indices), max_seq_length, self.vector.dim)
            for idx, seq in enumerate(seqs):
                if len(seq) != 0:
                    vector[idx, :len(seq), :] = torch.cat(seq, dim=0)
                lengths.append(len(seq))

        return vector, lengths

    def _encode_answer_vector_bert(self, answer):  # 向量求平均

        if isinstance(self.cached_answers_bert.get(answer, -1), int):
            answer_vec = torch.zeros(1, self.args.bert_dim)
            idk = self.stoi_bert.get(answer, -1)
            if idk >= 0:
                answer_vec = self.map_bert[idk]
            self.cached_answers_bert[answer] = answer_vec
        return self.cached_answers_bert[answer]

    def _encode_answer_vector_gae(self, answer):  # 向量求平均
        if isinstance(self.cached_answers_gae.get(answer, -1), int):
            answer_vec = torch.zeros(1, self.vector.dim)
            idk = self.stoi_gae.get(answer, -1)
            if idk >= 0:
                answer_vec = self.map_gae[idk].reshape(1, 300)
            self.cached_answers_gae[answer] = answer_vec
        return self.cached_answers_gae[answer]

    def _encode_answer_vector_g2v(self, answer):  # 向量求平均
        if isinstance(self.cached_answers_g2v.get(answer, -1), int):
            answer_vec = torch.zeros(1, self.vector.dim)

            idk = self.stoi_kg.get(answer, -1)
            if idk >= 0:
                answer_vec = self.map_kg[idk].reshape(1, 300)
            self.cached_answers_g2v[answer] = answer_vec
        return self.cached_answers_g2v[answer]

    def _encode_answer_vector_w2v(self, answer):  # 向量求平均
        if isinstance(self.cached_answers_w2v.get(answer, -1), int):
            tokens = nltk.word_tokenize(answer)
            answer_vec = torch.zeros(1, self.vector.dim)
            cnt = 0
            for i, token in enumerate(tokens):
                if self.vector.check(token):
                    answer_vec += self.vector[token]
                    cnt += 1
            self.cached_answers_w2v[answer] = answer_vec / (cnt + 1e-12)
            # pdb.set_trace()
        return self.cached_answers_w2v[answer]

    def _encode_answer_sequence_w2v(self, answer):
        if isinstance(self.cached_answers_w2v.get(answer, -1), int):
            tokens = nltk.word_tokenize(answer)
            answer_seq = []
            for i, token in enumerate(tokens):
                if self.vector.check(token):
                    answer_seq.append(self.vector[token].view(1, self.vector.dim))
                else:
                    answer_seq.append(self.vector['<unk>'].view(1, self.vector.dim))
            self.cached_answers_w2v[answer] = answer_seq

        return self.cached_answers_w2v[answer]

    def _encode_multihot_labels(self, answers):
        """ Turn an answer into a vector """
        max_answer_index = self.args.TEST.max_answer_index
        answer_vec = torch.zeros(max_answer_index)
        for answer in answers:
            index = self.answer_to_index.get(answer)
            if index is not None:
                if index < max_answer_index:
                    answer_vec[index] += 1
        return answer_vec

    def evaluate(self, predictions):
        raise NotImplementedError
