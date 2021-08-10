import os.path as osp
import numpy as np
import random
import torch
from easydict import EasyDict as edict
import argparse
import pdb


class cfg():
    def __init__(self):

        self.fusion_model_path = ""
        self.answer_net_path = ""

        self.joint_test_way = 0

        self.this_dir = osp.dirname(__file__)
        self.data_root = osp.abspath(osp.join(self.this_dir, '..', '..', 'data', 'KG_VQA'))
        self.this_dir = osp.dirname(__file__)
        self.project_root = osp.abspath(osp.join(self.this_dir, '..'))
        self.method_choice = "KG"
        self.ans_fusion = 'RNN_concate'
        self.fusion_model = ''
        self.requires_grad = 1
        self.bert_dim = 1024
        self.KGE = "TransE"
        self.KGE_init = None  # none or w2v
        self.glimpse = 4
        self.ans_feature_len = 0
        self.patience = 30
        self.v_dim = 2048

        self.FVQA = edict()

        # FVQA params

        self.FVQA.max_ans = 500
        self.FVQA.data_choice = "0"

        self.FVQA.entity_num = "all"
        self.FVQA.data_path = osp.join(self.data_root, "fvqa")

        self.FVQA.exp_data_path = osp.join(self.FVQA.data_path, "exp_data")
        self.FVQA.common_data_path = osp.join(self.FVQA.exp_data_path, "common_data")
        self.FVQA.test_data_path = osp.join(self.FVQA.exp_data_path, "test_data")
        self.FVQA.train_data_path = osp.join(self.FVQA.exp_data_path, "train_data")
        self.FVQA.seen_train_data_path = osp.join(self.FVQA.exp_data_path, "train_seen_data")
        self.FVQA.unseen_test_data_path = osp.join(self.FVQA.exp_data_path, "test_unseen_data")
        self.FVQA.seen_test_data_path = osp.join(self.FVQA.exp_data_path, "test_seen_data")
        self.FVQA.model_save_path = osp.join(self.FVQA.data_path, "model_save")
        self.FVQA.runs_path = osp.join(self.FVQA.data_path, "model_save")

        self.FVQA.qa_path = self.FVQA.exp_data_path
        self.FVQA.feature_path = osp.join(self.FVQA.common_data_path, 'fvqa-resnet-14x14.h5')
        self.FVQA.answer_vocab_path = osp.join(
            self.FVQA.common_data_path, 'answer.vocab.fvqa.' + str(self.FVQA.max_ans) + '.json')
        self.FVQA.fact_vocab_path = osp.join(self.FVQA.common_data_path, 'answer.vocab.fvqa.fact.500.json')
        self.FVQA.relation_vocab_path = osp.join(self.FVQA.common_data_path, 'answer.vocab.fvqa.relation.500.json')

        self.FVQA.fact_relation_to_ans_path = osp.join(self.FVQA.common_data_path, "fact_relation_dict.data")
        self.FVQA.img_path = osp.join(self.FVQA.qa_path, 'images')

        self.FVQA.kg_path = osp.join(self.FVQA.common_data_path, "KG_embedding")
        self.FVQA.gae_path = osp.join(self.FVQA.common_data_path, "GAE_embedding")
        self.FVQA.bert_path = osp.join(self.FVQA.common_data_path, "BERT_embedding")

        self.FVQA.gae_node_num = 3463
        self.FVQA.gae_init = "w2v"  # or w2v
        # 有问题
        # self.FVQA.qa = 'train2014'
        # self.FVQA.task = 'OpenEnded'
        # self.FVQA.dataset = 'mscoco'

        # self.dataset = self.FVQA

        self.cache_path = osp.join(self.data_root, '.cache')
        self.output_path = self.FVQA.model_save_path
        self.embedding_size = 1024  # embedding dimensionality
        self.hidden_size = 2 * self.embedding_size  # hidden embedding
        # a joint question vocab across all dataset
        self.question_vocab_path = osp.join(self.FVQA.common_data_path, 'question.vocab.json')  # 修改这里之后所有的预存文件（pt）都要删除

        # preprocess config
        self.image_size = 448
        self.output_size = self.image_size // 32
        self.preprocess_batch_size = 100  # 64
        self.output_features = 2048
        self.central_fraction = 0.875

        # Train params
        self.TRAIN = edict()
        self.TRAIN.epochs = 600
        self.TRAIN.batch_size = 128  # 128
        self.TRAIN.lr = 5e-4  # default Adam lr 1e-3
        self.TRAIN.lr_decay_step = 3
        self.TRAIN.lr_decay_rate = .70

        # self.TRAIN.data_workers = 20
        self.TRAIN.data_workers = 8  # 10
        self.TRAIN.answer_batch_size = self.FVQA.max_ans  # batch size for answer network
        self.TRAIN.max_negative_answer = self.FVQA.max_ans  # max negative answers to sample

        # Test params
        self.TEST = edict()
        self.TEST.batch_size = 128
        self.TEST.max_answer_index = self.FVQA.max_ans  # max answer index for computing acc   853

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu_id', default=1, type=int)
        parser.add_argument('--finetune', action='store_true')
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--max_ans', default=500, type=int)  # 3000 300##
        parser.add_argument('--loss_temperature', default=0.01, type=float)
        # parser.add_argument('--pretrained_model', default=None, type=str)
        parser.add_argument('--answer_embedding', default='MLP', choices=['RNN', 'MLP'])  # 答案编码：MLP or RNN##
        # parser.add_argument('--context_embedding', default='BoW', choices=['SAN', 'BoW'])  # Q I 内容编码：SAN or MLP
        parser.add_argument('--embedding_size', default=1024, choices=[1024, 300, 512], type=int)  # 答案编码：MLP or RNN##
        parser.add_argument('--epoch', default=800, type=int)  # 答案编码：MLP or RNN ##
        # choice model
        parser.add_argument('--fusion_model', default='SAN', choices=['MLP', 'SAN', 'UD', 'MUTAN', 'BAN', 'ViLBERT'])
        parser.add_argument('--requires_grad', default=0, type=int, choices=[0, 1])
        # choice class
        parser.add_argument('--method_choice', default='W2V',
                            choices=['CLS', 'W2V', 'KG', 'GAE', 'KG_W2V', 'KG_GAE', 'GAE_W2V', 'KG_GAE_W2V'])
        parser.add_argument('--ans_fusion', default='Simple_concate',
                            choices=['RNN_concate', 'GATE_attention', 'GATE', 'RNN_GATE_attention', 'Simple_concate'])
        # KG situation
        parser.add_argument('--KGE', default='TransE',
                            choices=['TransE', 'ComplEx', "TransR", "DistMult"])  # 答案编码：MLP or RNN ##
        parser.add_argument('--KGE_init', default="w2v")  # None  # none or w2v ##
        parser.add_argument('--GAE_init', default="random")  # None  # random or w2v ##
        parser.add_argument('--ZSL', type=int, default=0)  # None  # random or w2v ##
        parser.add_argument('--entity_num', default="all", choices=['all', '4302'])  # todo: 完成不同子图情况的... ##

        parser.add_argument('--data_choice', default='0', choices=['0', '1', '2', '3', '4'])
        parser.add_argument('--name', default=None, type=str)  # 定义名字后缀

        parser.add_argument("--no-tensorboard", default=False, action="store_true")
        parser.add_argument("--exp_name", default="", type=str, required=True, help="Experiment name")
        parser.add_argument("--dump_path", default="dump/", type=str, help="Experiment dump path")
        parser.add_argument("--exp_id", default="", type=str, help="Experiment ID")
        parser.add_argument("--random_seed", default=4567, type=int)
        parser.add_argument("--freeze_w2v", default=1, type=int, choices=[0, 1])
        parser.add_argument("--ans_net_lay", default=0, type=int, choices=[0, 1, 2])
        parser.add_argument("--fact_map", default=0, type=int, choices=[0, 1])
        parser.add_argument("--relation_map", default=0, type=int, choices=[0, 1])

        parser.add_argument("--now_test", default=0, type=int, choices=[0, 1])
        parser.add_argument("--save_model", default=0, type=int, choices=[0, 1])

        parser.add_argument("--joint_test_way", default=0, type=int, choices=[0, 1])
        parser.add_argument("--top_rel", default=10, type=int)
        parser.add_argument("--top_fact", default=100, type=int)
        parser.add_argument("--soft_score", default=10, type=int)  # 10 or 10000
        parser.add_argument("--mrr", default=0, type=int)
        args = parser.parse_args()
        return args

    def update_train_configs(self, args):
        self.gpu_id = args.gpu_id
        self.finetune = args.finetune
        self.answer_embedding = args.answer_embedding
        self.name = args.name
        self.no_tensorboard = args.no_tensorboard
        self.exp_name = args.exp_name
        self.dump_path = args.dump_path
        self.exp_id = args.exp_id
        self.random_seed = args.random_seed
        self.freeze_w2v = args.freeze_w2v
        self.loss_temperature = args.loss_temperature
        self.ZSL = args.ZSL
        self.ans_net_lay = args.ans_net_lay
        self.fact_map = args.fact_map
        self.relation_map = args.relation_map
        self.now_test = args.now_test
        self.save_model = args.save_model
        self.joint_test_way = args.joint_test_way
        self.top_rel = args.top_rel
        self.top_fact = args.top_fact
        self.soft_score = args.soft_score
        self.mrr = args.mrr

        if args.ZSL == 1:
            print("ZSL setting...")
            self.FVQA.test_data_path = self.FVQA.unseen_test_data_path
            self.FVQA.train_data_path = self.FVQA.seen_train_data_path

        if args.fusion_model == 'UD' or args.fusion_model == 'BAN':
            self.FVQA.feature_path = osp.join(self.FVQA.common_data_path, 'fvqa_36.hdf5')
            self.FVQA.img_id2idx = osp.join(self.FVQA.common_data_path, 'fvqa36_imgid2idx.pkl')
        self.requires_grad = True if args.requires_grad == 1 else False
        self.fusion_model = args.fusion_model
        self.TRAIN.batch_size = args.batch_size
        # self.TRAIN.answer_batch_size = args.answer_batch_size
        self.method_choice = args.method_choice
        self.ans_fusion = args.ans_fusion
        self.embedding_size = args.embedding_size
        self.FVQA.data_choice = args.data_choice
        self.FVQA.max_ans = args.max_ans
        self.TRAIN.epochs = args.epoch
        self.FVQA.KGE = args.KGE
        self.FVQA.KGE_init = args.KGE_init
        self.FVQA.gae_init = args.GAE_init
        self.FVQA.entity_num = args.entity_num

        if self.fact_map:
            self.FVQA.max_ans = 2791
        if self.relation_map:
            self.FVQA.max_ans = 103

        self.TEST.max_answer_index = self.FVQA.max_ans
        self.TRAIN.answer_batch_size = self.FVQA.max_ans  # batch size for answer network
        self.TRAIN.max_negative_answer = self.FVQA.max_ans

        self.FVQA.answer_vocab_path = osp.join(
            self.FVQA.common_data_path, 'answer.vocab.fvqa.' + str(self.FVQA.max_ans) + '.json')

        if "KG" in self.method_choice:
            self.FVQA.relation2id_path = osp.join(self.FVQA.kg_path, "relations_" + self.FVQA.entity_num + ".tsv")
            self.FVQA.entity2id_path = osp.join(self.FVQA.kg_path, "entities_" + self.FVQA.entity_num + ".tsv")
            if self.KGE_init != "w2v":
                self.FVQA.entity_path = osp.join(self.FVQA.kg_path, "fvqa_" +
                                                 self.FVQA.entity_num + "_" + self.KGE + "_entity.npy")
                self.FVQA.relation_path = osp.join(self.FVQA.kg_path, "fvqa_" +
                                                   self.FVQA.entity_num + "_" + self.KGE + "_relation.npy")
            else:
                self.FVQA.entity_path = osp.join(self.FVQA.kg_path, "fvqa_" +
                                                 self.FVQA.entity_num + "_w2v_" + self.KGE + "_entity.npy")
                self.FVQA.relation_path = osp.join(self.FVQA.kg_path, "fvqa_" +
                                                   self.FVQA.entity_num + "_w2v_" + self.KGE + "_relation.npy")
