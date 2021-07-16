import json
import os
import os.path as osp
import nltk
from collections import Counter
import torch
import torch.utils.data as data
import pdb

################
from .base import VisualQA
from .preprocess import process_punctuation


def get_loader(args, vector, train=False, val=False):
    """ Returns a data loader for the desired split """
    assert train + val == 1, 'need to set exactly one of {train, val, test} to True'  # 必须有一个为真
    id = args.FVQA.data_choice
    if train:
        filepath = "train" + id
        print("use train data:", id)
        filepath = os.path.join(args.FVQA.train_data_path, filepath)
    else:
        filepath = "test" + id
        filepath = os.path.join(args.FVQA.test_data_path, filepath)

    split = FVQA(  # 定义每一次训练的VQA输入 # ok
        args,
        path_for(args, train=train, val=val, filepath=filepath),  # train的问题
        vector,  # 对应的词向量
        file_path=filepath
    )
    batch_size = args.TRAIN.batch_size
    if val:
        batch_size = args.TEST.batch_size
    loader = torch.utils.data.DataLoader(  # 定义传统的DataLoader
        split,
        batch_size=batch_size,
        shuffle=True,  # only shuffle the data in training
        pin_memory=True,
        num_workers=args.TRAIN.data_workers,
    )

    return loader


class FVQA(VisualQA):  # ok
    """ FVQA dataset, open-ended """

    def __init__(self, args, qa_path, vector, file_path=None):
        self.args = args
        answer_vocab_path = self.args.FVQA.answer_vocab_path
        super(FVQA, self).__init__(args, vector)
        # load annotation
        with open(qa_path, 'r') as fd:
            self.qa_json = json.load(fd)

        # print('extracting answers...')

        # 把问题变成id向量+长度的表示, 答案变成id向量
        if args.fact_map:
            #  得到对应的名字
            name = "fact"
            self.answers = list(prepare_fact(self.qa_json))  # 候选答案列表的列表 [[answer1,answer2,...][....]] 每个问题对应的答案. 单词表示
        elif args.relation_map:
            name = "relation"
            self.answers = list(prepare_relation(self.qa_json))  # 候选答案列表的列表 [[answer1,answer2,...][....]] 每个问题对应的答案. 单词表示
        else:
            name = "answer"
            self.answers = list(prepare_answers(self.qa_json))  # 候选答案列表的列表 [[answer1,answer2,...][....]] 每个问题对应的答案. 单词表示

        cache_filepath = self._get_cache_path(qa_path, file_path, name)

        # self.support_relation = list(prepare_relation(self.qa_json))
        self.questions, self.answer_indices = self._qa_id_represent(cache_filepath)
        # pdb.set_trace()
        # process images 处理图片

    def open_hdf5(self):
        self.image_features_path = self.args.FVQA.feature_path
        self.image_id_to_index = self._create_image_id_to_index()  # 得到图片编号到下标的表示
        # self.image_ids = [q['image_id'] for q in questions_json['questions']]
        self.image_ids = self._get_img_id()

    def __getitem__(self, item):  # ok
        if not hasattr(self, 'image_ids'):
            self.open_hdf5()
        # if item > len(self.answers):
        #     pdb.set_trace()

        question, question_length = self.questions[item]  # 问题向量列表
        # sample answers
        # self.answer_indices[item]：[1,2,3] or [-1, -1 ...]
        # answer_cands = Counter(self.answer_indices[item])  # 单个答案 返回类型：Counter({1: 1, 2: 1, 3: 1})
        # answer_indices = list(answer_cands.keys())  # 答案有哪几个（下标）[[1,2,3]]
        # counts = list(answer_cands.values())  # 这几个答案分别出现了多少次[10]

        label = self._encode_multihot_labels(self.answers[item])  # 答案的multihot表示 前百分之多少的答案
        image_id = self.image_ids[item]
        image, spa = self._load_image(image_id)  # 直接获得图片的特征
        # unique_answers, answer_vectors = self._generate_batch_answer(answer_indices, counts)
        # answer_vectors == label
        # assert answer_vectors == label
        # return image, spa, question, unique_answers, answer_vectors, label, item, question_length
        # pdb.set_trace()
        return image, spa, question, label, item, question_length

    def _get_cache_path(self, qa_path, file_path, name):
        w2v = ""
        if "KG" in self.args.method_choice:
            if "w2v" in self.args.FVQA.entity_path:
                w2v = "(w2vinit)_" + self.args.FVQA.entity_num + "_" + self.args.FVQA.KGE
            else:
                w2v = "_" + self.args.FVQA.entity_num + "_" + self.args.FVQA.KGE
        if "train" in qa_path:
            cache_filepath = osp.join(file_path, "fvqa_" + name + "_and_question_train_" +
                                      self.args.method_choice + w2v + "_" + str(self.args.FVQA.max_ans) + ".pt")
        else:
            cache_filepath = osp.join(file_path, "fvqa_" + name + "_and_question_test_" + self.args.method_choice + w2v + "_" + str(
                self.args.FVQA.max_ans) + ".pt")
        return cache_filepath

    def _qa_id_represent(self, cache_filepath):
        if not os.path.exists(cache_filepath):
            # print('encoding questions...')
            questions = list(prepare_questions(self.qa_json))  # 问题词列表的列表
            questions = [self._encode_question(q) for q in questions]  # 把问题变成id向量+长度的表示

            # 对于候选答案列表中的每一个问题对应的候选答案列表，转换成下标表示[[1,2,3],[2,3,4]......]  1——>一个答案
            answer_indices = [[self.answer_to_index.get(_a, -1) for _a in a] for a in self.answers]  # 如果没有匹配就是 -1
            torch.save({'questions': questions, 'answer_indices': answer_indices}, cache_filepath)

        else:
            # 已经有，对应这个训练/测试集 的问题w2v表，[train 和 test是不一样的]
            _cache = torch.load(cache_filepath)
            questions = _cache['questions']  # 词向量列表 + 长度
            answer_indices = _cache['answer_indices']  # 答案下标
            # self.answer_vectors = _cache['answer_vectors']  # 答案的向量表示[平均]

        return questions, answer_indices

    def _get_img_id(self):
        image_ids = []
        keys = list(self.qa_json.keys())
        for a in keys:
            filename = self.qa_json[a]["img_file"]
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            if not filename.endswith('.jpg'):
                id += 1000000  # 把jpg和jpeg的分开
                # pdb.set_trace()
            image_ids.append(id)
        return image_ids

    # def _generate_batch_answer(self, indices, counts):  # 获得每一个batch的500个候选答案。
    #     unique_answers = list(range(0, self.args.FVQA.max_ans))
    #     # unique_answers = list(set( aid for aids in indices for aid in aids ))
    #     answer_dict = {k: i for i, k in enumerate(unique_answers)}
    #     answer_vector = torch.zeros(len(indices), len(unique_answers))  # 128,500
    #
    #     for i in range(len(counts)):  # 128
    #         for j, c in zip(indices[i], counts[i]):
    #             answer_vector[i, answer_dict[j]] = c  # 把出现的次数附上
    #
    #     return unique_answers, answer_vector


def path_for(args, train=False, val=False, filepath=""):
    # tra = "all_qs_dict_release_train_" + str(args.FVQA.max_ans) + ".json"
    # tes = "all_qs_dict_release_test_" + str(args.FVQA.max_ans) + ".json"
    tra = "all_qs_dict_release_train_500.json"
    tes = "all_qs_dict_release_test_500.json"
    if train == True:
        return os.path.join(args.FVQA.train_data_path, filepath, tra)
    else:
        return os.path.join(args.FVQA.test_data_path, filepath, tes)


def prepare_questions(questions_json):  # ok
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    keys = list(questions_json.keys())
    questions = []
    for a in keys:
        questions.append(questions_json[a]['question'])  # question的list
    for question in questions:
        question = question.lower()[:-1]
        yield nltk.word_tokenize(process_punctuation(question))  # 得到一个词的list，例如['I', 'LOVE', 'YOU']


def prepare_answers(answers_json):  # ok
    """ Normalize answers from a given answer json in the usual VQA format. """
    keys = list(answers_json.keys())
    answers = []

    for a in keys:
        answer = answers_json[a]["answer"]
        answers.append([answer] * 10)  # 双层list，内层的list对应一个问题的答案序列
    for answer_list in answers:
        ret = list(map(process_punctuation, answer_list))  # 去除标点等操作
        yield ret


def prepare_fact(answers_json):  # ok
    """ Normalize answers from a given answer json in the usual VQA format. """
    keys = list(answers_json.keys())
    support_facts = []
    for a in keys:
        answer = answers_json[a]["answer"]
        facts = answers_json[a]["fact"]
        f1 = facts[0]
        f2 = facts[2]
        if answer != f1 and answer != f2:
            pdb.set_trace()
        assert (answer == f1 or answer == f2)
        if answer == f1:
            fact = f2
        else:
            fact = f1
        support_facts.append([fact] * 10)  # 双层list，内层的list对应一个问题的答案序列
    for support_facts_list in support_facts:
        ret = list(map(process_punctuation, support_facts_list))  # 去除标点等操作
        yield ret


def prepare_relation(answers_json):  # ok
    """ Normalize answers from a given answer json in the usual VQA format. """
    keys = list(answers_json.keys())
    relations = []
    for a in keys:
        facts = answers_json[a]["fact"]
        relation = facts[1]

        relations.append([relation] * 10)  # 双层list，内层的list对应一个问题的答案序列
    for relation_list in relations:
        ret = list(map(process_punctuation, relation_list))  # 去除标点等操作
        yield ret
