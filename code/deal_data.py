# Data cleaning about the data in F-VQA / ZS-F-VQA
##
from config import cfg
import os.path as osp
import pickle
import json
import pdb
import re
from utils import dele_a, transfer, hand_remove, deal_fact
from collections import defaultdict
import tqdm
import Levenshtein
import wordninja
from data import fvqa, preprocess
import random
import numpy as np
from collections import defaultdict


class Runner:
    def __init__(self, args):
        self.args = args
        self.path = osp.join(args.data_root, "data", "FVQA/")
        self.data_path = osp.join(self.path, "new_dataset_release")
        self.split_path = osp.join(self.path, "Name_Lists")
        self.exp_data = osp.join(args.data_root, "fvqa", "exp_data")
        self.e1_list = []
        self.r_list = []
        self.e2_list = []
        self.entity_list = []
        # 这个entity 在哪些VQA pair中出现过。
        self.e1_show_key = defaultdict(list)
        self.e2_show_key = defaultdict(list)
        self.all_entity = []

    # 得到一个所有fact都直接被包含在内的 json文件（不需要跳转）
    def get_new_all_json(self):
        path = osp.join(self.data_path, "all_qs_dict_release_combine_all.json")

        if not osp.exists(path):
            with open(osp.join(self.data_path, "all_fact_triples_release.json"), "r", encoding='utf8') as ffp:
                dic_all = json.load(ffp)
                # pdb.set_trace()
                for i in dic_all.keys():
                    # fact_source = dic[i]["fact"][0]
                    fact = dic_all[i]
                    fact['e1'] = deal_fact(dic_all[i], fact['e1'])
                    fact['e2'] = deal_fact(dic_all[i], fact['e2'])
                    dic_all[i]["fact"] = []
                    dic_all[i]["fact"].append(fact['e1'])
                    dic_all[i]["fact"].append(fact['r'].split('/')[-1])
                    dic_all[i]["fact"].append(fact['e2'])
                    # pdb.set_trace()
                    del dic_all[i]['KB']
                    del dic_all[i]['e1_label']
                    # del dic_all[i]['uri']
                    del dic_all[i]['e2_label']
                    # del dic_all[i]['sources']
                    # del dic_all[i]['context']
                    del dic_all[i]['score']
        else:
            # 需要人工去噪
            with open(path, 'w') as fd:
                json.dump(dic_all, fd)
                print("get_new_json_combile done!（remember to do some human check !!!）")

    # 得到一个所有fact都直接被包含在内的 json文件（不需要跳转）
    def get_new_json(self):
        path = osp.join(self.data_path, "all_qs_dict_release_combine.json")
        if not osp.exists(path):
            with open(osp.join(self.data_path, "all_qs_dict_release_cp.json"), "r") as fp:
                dic = json.load(fp)
                with open(osp.join(self.data_path, "all_fact_triples_release.json"), "r", encoding='utf8') as ffp:
                    dic_all = json.load(ffp)
                    # pdb.set_trace()
                    for i in dic_all.keys():
                        fact_source = dic[i]["fact"][0]
                        fact = dic_all[fact_source]
                        fact['e1'] = deal_fact(dic[i], fact['e1'])
                        fact['e2'] = deal_fact(dic[i], fact['e2'])
                        dic[i]["fact"][0] = fact['e1']
                        dic[i]["fact"].append(fact['r'].split('/')[-1])
                        dic[i]["fact"].append(fact['e2'])
                        del dic[i]['ans_source']
                        del dic[i]['visual_concept']
            # 需要人工去噪
            with open(path, 'w') as fd:
                json.dump(dic, fd)
                print("get_new_json done!（remember to do some human check !!!）")

    def get_entity_filter(self):
        # 把头尾实体筛选一遍，并且储存

        with open(osp.join(self.data_path, "all_qs_dict_release_combine.json"), 'r') as fp:
            dic = json.load(fp)
            for i in dic.keys():
                for j in [0, 1, 2]:
                    dic[i]["fact"][j] = dic[i]["fact"][j].lower().replace("  ", " ")
                    if dic[i]["fact"][j][0] == " ":
                        dic[i]["fact"][j] = dic[i]["fact"][j][1:]
                    if len(dic[i]["fact"][j]) > 2 and dic[i]["fact"][j][-2] == "#":
                        dic[i]["fact"][j] = dic[i]["fact"][j][:-2]

                self.e1_list.append(dic[i]["fact"][0])
                self.r_list.append(dic[i]["fact"][1])
                self.e2_list.append(dic[i]["fact"][2])
                self.e1_show_key[dic[i]["fact"][0]].append(i)
                self.e2_show_key[dic[i]["fact"][2]].append(i)
            self.entity_list = set(self.e1_list + self.e2_list)
            self.entity_list = list(self.entity_list)
            self.r_list = list(set(self.r_list))
            # pdb.set_trace()
            print("get_entity_filter done!")

    def get_all_entity(self):
        path = osp.join(self.data_path, 'ids_new.data')
        if not osp.exists(path):

            # 得到所有的头尾实体，并且排序
            entity_list = []
            with open(osp.join(self.data_path, "FVQA_triple_new_2.txt"), 'r', encoding='utf-8') as f:
                # k = 0
                while 1:
                    line = f.readline()
                    if not line:
                        break
                    if line[:3] == '***':
                        continue
                    # k += 1
                    # if k % 1000 == 0:
                    #     print(k, len(lis))
                    line = re.split('\t|\n', line)
                    entity_list.append(line[0].lower().replace("-", " "))
                    entity_list.append(line[2].lower().replace("-", " "))
            entity_set = set(entity_list)

            def rule_4(a):
                return entity_list.count(a)

            entity_sort = list(set(entity_list))
            entity_sort.sort(key=rule_4, reverse=True)

            with open(path, 'wb') as f:
                pickle.dump(entity_sort, f)

        else:
            with open(path, 'rb') as f:  # 按出现数量排序过了的实体
                print("load ids_new.data")
                entity_sort = pickle.load(f)

        entity_sort.remove('y')
        entity_sort.remove('and')
        entity_sort.remove('yes')
        entity_sort.remove('no')

        path = osp.join(self.data_path, "all_qs_dict_release_combine_filter.json")

        if not osp.exists(path):
            with open(osp.join(self.data_path, "all_qs_dict_release_combine.json"), 'r') as fp:
                dic = json.load(fp)
                Noin = []
                for entity in tqdm.tqdm(self.entity_list):
                    entity_orig = entity
                    entity = entity.replace("_", " ").replace("-", " ")
                    entity = entity.replace("Category:", "").replace("category:", "")
                    entity = entity.replace("(", "").replace(")", "")
                    entity_list = [entity]

                    dele_a_list = dele_a(entity)
                    transfer_a = [transfer(entity)]
                    # entity_list.append(no_)
                    entity_list = entity_list + transfer_a  # 变形
                    entity_list = entity_list + dele_a_list  # 去冠词
                    entity_list = entity_list + dele_a(transfer_a[0])  # 变形后去冠词
                    for i in dele_a_list:
                        entity_list = entity_list + [transfer(i)]  # 去冠词后变形

                    entity_list = list(set(entity_list))
                    hand_list = []
                    for k in entity_list:
                        hand_list = hand_list + hand_remove(k)  # 手动去特殊形式
                    entity_list = entity_list + list(set(hand_list))
                    entity_list = list(set(entity_list))
                    flag = 0

                    # print("change entity...")
                    for key in entity_sort:
                        if key in entity_list:
                            flag = 1
                            self.all_entity.append(key)
                            for j in self.e1_show_key[entity_orig]:  # 答案是这个的编号
                                dic[j]['fact'][0] = key
                            for j in self.e2_show_key[entity_orig]:  # 答案是这个的编号
                                dic[j]['fact'][2] = key
                            break
                    if flag:
                        continue

                    Noin.append(entity_orig)
            print("all entity num:", len(list(set(self.all_entity))))
            print("no in :", Noin)
            print("no in num :", len(Noin))

            # entity 筛选过的。此时答案和entity 统一了

            with open(path, 'w') as fp:
                json.dump(dic, fp)
            print("get_all_entity filter done!")

    def fusion_answer_and_entity(self):
        # 把答案里面出现的entity对齐到entity中。
        # 使用编辑距离
        path = osp.join(self.data_path, "all_qs_dict_release_combine_filter_fusion.json")
        if not osp.exists(path):
            with open(osp.join(self.data_path, "all_qs_dict_release_combine_filter.json"), 'r') as fp:
                dic = json.load(fp)
                with open(osp.join(self.data_path, "ans_entity_map.txt"), 'w') as ffp:
                    for key in dic.keys():
                        strout = "not match: "
                        e1 = dic[key]["fact"][0]
                        e2 = dic[key]["fact"][2]
                        ans = dic[key]["answer"]
                        # 和头实体相似度大于尾实体
                        if Levenshtein.ratio(ans, e1) > Levenshtein.ratio(ans, e2):
                            strout += dic[key]["fact"][2]
                            strout += "\t\t\t\t\t  match: "
                            strout += dic[key]["fact"][0]
                            strout += " -> "
                            strout += ans
                            dic[key]["fact"][0] = ans

                        else:
                            strout += dic[key]["fact"][0]
                            strout += "\t\t\t\t\t  match: "
                            strout += dic[key]["fact"][2]
                            strout += " -> "
                            strout += ans
                            dic[key]["fact"][2] = ans
                        ffp.write(strout + "\n")

                    print("fusion_answer_and_entity done!")

                with open(osp.join(self.data_path, "all_qs_dict_release_combine_filter_fusion.json"), 'w') as fp:
                    json.dump(dic, fp)

    def statistics_of_ans_and_entity(self, name=None, path=None):
        # 数据统计
        if path == None:
            path = osp.join(self.data_path, name)

        with open(path, 'r') as fp:
            dic = json.load(fp)
            ans_set = set()
            entity_set = set()
            relation_set = set()
            dic_len = 0
            for key in dic.keys():
                dic_len += 1
                e1 = dic[key]["fact"][0]
                r = dic[key]["fact"][1]
                e2 = dic[key]["fact"][2]
                ans = dic[key]["answer"]
                ans_set.add(ans)
                entity_set.add(e1)
                entity_set.add(e2)
                relation_set.add(r)

            ans_or_entity = ans_set | entity_set
            ans_and_entity = ans_set & entity_set
            print("ans_set len:", len(ans_set))
            print("entity_set len:", len(entity_set))
            print("ans_or_entity len:", len(ans_or_entity))
            print("ans_and_entity len:", len(ans_and_entity))
            print("relation len:", len(relation_set))
            print("dic len:", dic_len)

    def filter_top500_IQA_pair(self):
        # read ans file
        # store the map from id to ans (with dic)
        # TODO: optimize the code with matrix
        path = osp.join(self.data_path, "all_qs_dict_release_combine_filter_fusion_500.json")
        if not osp.exists(path):
            ans_2_id = {}
            with open(osp.join(self.data_path, "ans.txt"), 'r', encoding='utf-8') as f:
                while 1:
                    line = f.readline()
                    if not line:
                        break
                    line = re.split('-|\n', line)
                    ans_2_id[line[1]] = int(line[0])
            print(len(ans_2_id))
            with open(osp.join(self.data_path, "all_qs_dict_release_combine_filter_fusion.json"), 'r') as fp:
                dic = json.load(fp)
                dic_500 = {key: value for key, value in dic.items() if
                           dic[key]["answer"] in ans_2_id.keys() and ans_2_id[dic[key]["answer"]] <= 500}

            with open(path, 'w') as fp:
                json.dump(dic_500, fp)
                print("filter_top500_IQA_pair done!")

    def deal_relation(self):
        path = osp.join(self.data_path, "all_qs_dict_release_combine_filter_fusion_500.json")
        if not osp.exists(path):
            with open(osp.join(self.data_path, "all_qs_dict_release_combine_filter_fusion_500.json"), 'r') as fp:
                dic = json.load(fp)
            relation_set = set()
            for key in dic.keys():
                relation_set.add(dic[key]["fact"][1])

            print("relation len:", len(relation_set))
            relation_map = {}
            for relation in list(relation_set):
                relation_orig = relation
                # 是否需要把关系去掉？
                if relation[-2] == "#":
                    relation = relation[:-2]

                relation_split = wordninja.split(relation)
                for i in range(len(relation_split)):
                    relation_split[i] = relation_split[i].lower()

                if relation == "transnbhd":
                    relation_map[relation_orig] = "belong to"
                else:
                    relation_map[relation_orig] = ' '.join(relation_split)

            print(relation_map)
            for key in dic.keys():
                tmp = dic[key]["fact"][1]
                dic[key]["fact"][1] = relation_map[tmp]

            with open(path, 'w') as fp:
                json.dump(dic, fp)
                print("deal_relation done!")

    def split_data(self):
        # 把数据集划分出来
        for i in range(0, 5):
            num = str(i)
            train_name = osp.join(self.args.FVQA.train_data_path, "train" + num, "all_qs_dict_release_train_500.json")
            test_name = osp.join(self.args.FVQA.test_data_path, "test" + num, "all_qs_dict_release_test_500.json")

            if osp.exists(train_name) and osp.exists(test_name):
                continue

            img_train = []
            img_test = []

            with open(osp.join(self.split_path, "train_list_" + num + ".txt"), "r") as f:
                while 1:
                    line = f.readline()
                    if not line:
                        break
                    line = re.split('\n', line)
                    img_train.append(line[0])

            with open(osp.join(self.split_path, "test_list_" + num + ".txt"), "r") as f:
                while 1:
                    line = f.readline()
                    if not line:
                        break
                    line = re.split('\n', line)
                    img_test.append(line[0])

            with open(osp.join(self.data_path, "all_qs_dict_release_combine_filter_fusion_500.json"), 'r') as fp:
                dic = json.load(fp)

                dic_train = {key: value for key, value in dic.items() if dic[key]["img_file"] in img_train}
                dic_test = {key: value for key, value in dic.items() if dic[key]["img_file"] in img_test}

            # train_name = osp.join(cfg.FVQA.train_data_path, "train" + num, "all_qs_dict_release_train_500.json")
            # test_name = osp.join(cfg.FVQA.test_data_path, "test" + num, "all_qs_dict_release_test_500.json")
            ans_train = []
            ans_test = []
            q_train = []
            q_test = []
            for key, value in dic_train.items():
                ans_train.append(dic_train[key]["answer"])
                q_train.append(dic_train[key]["question"])
            for key, value in dic_test.items():
                ans_test.append(dic_test[key]["answer"])
                q_test.append(dic_test[key]["question"])

            ans_train_set = set(ans_train)
            q_train_set = set(q_train)
            ans_test_set = set(ans_test)
            q_test_set = set(q_test)

            with open(train_name, "w") as ff:
                json.dump(dic_train, ff)
                print("save to:", train_name)

            with open(test_name, "w") as ff:
                json.dump(dic_test, ff)
                print("save to:", test_name)
            # ans_set len: 387
            # entity_set len: 1842
            # ans_or_entity len: 1842
            # ans_and_entity len: 387
            # relation len: 71
            # dic len: 2669
            print(num, " train :", len(dic_train), len(ans_train_set), len(q_train_set))
            self.statistics_of_ans_and_entity(train_name)

            # ans_set len: 403
            # entity_set len: 1958
            # ans_or_entity len: 1958
            # ans_and_entity len: 403
            # relation len: 87
            # dic len: 2823
            print(num, " test :", len(dic_test), len(ans_test_set), len(q_test_set))
            self.statistics_of_ans_and_entity(test_name)
            print("dataset " + num + " done!")

    def preprocess_answer(self):
        pass

    def preprocess_fact(self):
        output_format = osp.join(self.args.FVQA.common_data_path, "answer.vocab.fvqa.fact.500.json")
        if not osp.exists(output_format):
            num = 2
            vqa_train_questions = osp.join(self.args.FVQA.train_data_path, "train" + str(num), "all_qs_dict_release_train_500.json")
            vqa_val_questions = osp.join(self.args.FVQA.test_data_path, "test" + str(num), "all_qs_dict_release_test_500.json")
            with open(vqa_train_questions, 'r') as fd:
                qaq1 = json.load(fd)
            with open(vqa_val_questions, 'r') as fd:
                qaq2 = json.load(fd)

            annotations = {**qaq1, **qaq2}
            # word2vec = Vector()
            facts = fvqa.prepare_fact(annotations)
            fact_vocab = preprocess.extract_vocab(facts, top_k=None)
            vocabs = {'answer': fact_vocab}
            print('* Dump output vocab to: {}'.format(output_format))
            with open(output_format, 'w') as fd:
                json.dump(vocabs, fd)
        print("preprocess_fact done!")

    def preprocess_relation(self):
        output_format = osp.join(self.args.FVQA.common_data_path, "answer.vocab.fvqa.relation.500.json")
        if not osp.exists(output_format):
            num = 2
            vqa_train_questions = osp.join(self.args.FVQA.train_data_path, "train" + str(num), "all_qs_dict_release_train_500.json")
            vqa_val_questions = osp.join(self.args.FVQA.test_data_path, "test" + str(num), "all_qs_dict_release_test_500.json")
            with open(vqa_train_questions, 'r') as fd:
                qaq1 = json.load(fd)
            with open(vqa_val_questions, 'r') as fd:
                qaq2 = json.load(fd)

            annotations = {**qaq1, **qaq2}
            # word2vec = Vector()
            relations = fvqa.prepare_relation(annotations)
            relation_vocab = preprocess.extract_vocab(relations, top_k=None)
            vocabs = {'answer': relation_vocab}
            print('* Dump output vocab to: {}'.format(output_format))
            with open(output_format, 'w') as fd:
                json.dump(vocabs, fd)
        print("preprocess_relation done!")

    def split_unseen_data(self):
        ans_2_id = {}

        with open(osp.join(self.data_path, "ans.txt"), 'r', encoding='utf-8') as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                line = re.split('-|\n', line)
                ans_2_id[line[1]] = int(line[0])
        print(len(ans_2_id))
        num = "0"
        img = []
        with open(osp.join(self.split_path, "train_list_" + num + ".txt"), "r") as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                line = re.split('\n', line)
                img.append(line[0])

        with open(osp.join(self.split_path, "test_list_" + num + ".txt"), "r") as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                line = re.split('\n', line)
                img.append(line[0])

        with open(osp.join(self.data_path, "all_qs_dict_release_combine_filter_fusion_500.json"), 'r') as fp:
            dic = json.load(fp)

        dic_all = {key: value for key, value in dic.items() if dic[key]["img_file"] in img}
        ans_id = list(range(1, 501))

        # split_unseen_data
        for i in range(5):
            num = str(i)
            train_name = osp.join(self.args.FVQA.seen_train_data_path, "train" + num, "all_qs_dict_release_train_500.json")
            test_name = osp.join(self.args.FVQA.unseen_test_data_path, "test" + num, "all_qs_dict_release_test_500.json")

            if not(osp.exists(train_name) and osp.exists(train_name)):
                ans_seen = random.sample(ans_id, 250)
                ans_unseen = list(set(ans_id) - set(ans_seen))

                dic_seen = {key: value for key, value in dic_all.items() if ans_2_id[dic[key]["answer"]] in ans_seen}
                dic_unseen = {key: value for key, value in dic_all.items() if ans_2_id[dic[key]["answer"]] in ans_unseen}

                with open(train_name, "w") as ff:
                    json.dump(dic_seen, ff)
                with open(test_name, "w") as ff:
                    json.dump(dic_unseen, ff)

                self.statistics_of_ans_and_entity(train_name)

                self.statistics_of_ans_and_entity(test_name)

                print("dataset " + num + " done!")

    def get_fact_relation_matrix(self):
        if not osp.exists(self.args.FVQA.fact_relation_to_ans_path):

            answer_vocab_path = self.args.FVQA.answer_vocab_path
            fact_vocab_path = self.args.FVQA.fact_vocab_path
            relation_vocab_path = self.args.FVQA.relation_vocab_path

            with open(fact_vocab_path, 'r') as fd:
                fact_vocab = json.load(fd)

            with open(relation_vocab_path, 'r') as fd:
                relation_vocab = json.load(fd)

            with open(answer_vocab_path, 'r') as fd:
                answer_vocab = json.load(fd)

            self.answer_to_index = answer_vocab['answer']
            self.index_to_answer = preprocess.invert_dict(self.answer_to_index)
            self.fact_to_index = fact_vocab['answer']
            self.index_to_fact = preprocess.invert_dict(self.fact_to_index)
            self.relation_to_index = relation_vocab['answer']
            self.index_to_relation = preprocess.invert_dict(self.relation_to_index)

            output_format = osp.join(self.args.FVQA.common_data_path, "fact_relation_dict.data")
            vqa_train_questions = osp.join(self.args.FVQA.train_data_path, "train2", "all_qs_dict_release_train_500.json")
            vqa_val_questions = osp.join(self.args.FVQA.test_data_path, "test2", "all_qs_dict_release_test_500.json")
            with open(vqa_train_questions, 'r') as fd:
                qaq1 = json.load(fd)
            with open(vqa_val_questions, 'r') as fd:
                qaq2 = json.load(fd)

            annotations = {**qaq1, **qaq2}

            fact_num = len(self.fact_to_index)
            ans_num = len(self.answer_to_index)
            rel_num = len(self.relation_to_index)

            # fact_relation_matrix = - np.ones([fact_num,rel_num ], dtype = int)
            fact_relation_to_ans = defaultdict(list)

            keys = list(annotations.keys())

            for a in keys:
                answer = annotations[a]["answer"]
                facts = annotations[a]["fact"]
                f1 = facts[0]
                rel = facts[1]
                f2 = facts[2]
                assert (answer == f1 or answer == f2)
                if answer == f1:
                    fact = f2
                else:
                    fact = f1

                fact = preprocess.process_punctuation(fact)
                rel = preprocess.process_punctuation(rel)
                name = str(self.fact_to_index[fact]) + "-" + str(self.relation_to_index[rel])
                fact_relation_to_ans[name].append(self.answer_to_index[answer])

            with open(output_format, 'w') as fd:
                json.dump(fact_relation_to_ans, fd)
                print("dump done!")

        with open(self.args.FVQA.fact_relation_to_ans_path, 'r') as fd:
            fact_relation_to_ans = json.load(fd)

    def preprocess_json_in_order(self):
        num = "3"

        data_path = osp.join(self.exp_data, "test_data", "test" + num, "all_qs_dict_release_test_500.json")
        output_format = osp.join(self.exp_data, "test_data", "test" + num, "all_qs_dict_release_test_500_inorder.json")

        if not osp.exists(output_format):
            with open(data_path, 'r') as fd:
                annotations = json.load(fd)
            keys = list(annotations.keys())
            tmp = 0
            new_annotations = {}
            for a in keys:
                new_annotations[str(tmp)] = annotations[a]
                tmp += 1

            with open(output_format, 'w') as fd:
                json.dump(new_annotations, fd)
                print("dump done!")

    def disjoint_judge(self):
        fact_id_path = osp.join(self.args.FVQA.common_data_path, "answer.vocab.fvqa.fact.500.json")
        answer_id_path = osp.join(self.args.FVQA.common_data_path, "answer.vocab.fvqa.500.json")
        with open(fact_id_path, 'r') as fd:
            self.fact_id = json.load(fd)
            self.fact_id = self.fact_id['answer']
            list_fact = list(self.fact_id)
        with open(answer_id_path, 'r') as fd:
            self.answer_id = json.load(fd)
            self.answer_id = self.answer_id['answer']
            list_ans = list(self.answer_id)
        all = 0
        for i in list_ans:
            if i in list_fact:
                all += 1
        print(all)

    def data_analysis(self, name):
        if name == "zsl":
            testpath = "test_unseen_data"
            trainpath = "train_seen_data"
        else:
            testpath = "test_data"
            trainpath = "train_data"

        train_triplet_num = 0
        test_triplet_num = 0
        and_answer_num = 0
        and_entity_num = 0
        and_question_num = 0
        and_image_num = 0
        and_answer_class = 0
        and_entity_class = 0
        and_question_class = 0
        and_image_class = 0
        train_question_class = 0
        test_question_class = 0
        train_answer_class = 0
        test_answer_class = 0
        train_entity_class = 0
        test_entity_class = 0
        train_image_class = 0
        test_image_class = 0

        for num in range(5):
            test_question = []
            train_question = []
            test_answer = []
            test_image = []
            train_answer = []
            test_entity = []
            train_entity = []
            train_image = []

            num = str(num)
            datapath_test = osp.join(self.exp_data, testpath, "test" + num, "all_qs_dict_release_test_500.json")
            datapath_train = osp.join(self.exp_data, trainpath, "train" + num, "all_qs_dict_release_train_500.json")

            with open(datapath_test, 'r') as fd:
                test_data = json.load(fd)
            with open(datapath_train, 'r') as fd:
                train_data = json.load(fd)

            test_data_keys = list(test_data.keys())
            train_data_keys = list(train_data.keys())

            for key in test_data_keys:
                test_question.append(test_data[key]["question"])
                test_answer.append(test_data[key]["answer"])
                test_image.append(test_data[key]["img_file"])
                e1 = test_data[key]["fact"][0]
                e2 = test_data[key]["fact"][2]
                ans = test_data[key]["answer"]
                # 和头实体相似度大于尾实体
                if Levenshtein.ratio(ans, e1) > Levenshtein.ratio(ans, e2):
                    test_entity.append(e2)
                else:
                    test_entity.append(e1)

            for key in train_data_keys:
                train_question.append(train_data[key]["question"])
                train_answer.append(train_data[key]["answer"])
                train_image.append(train_data[key]["img_file"])
                e1 = train_data[key]["fact"][0]
                e2 = train_data[key]["fact"][2]
                ans = train_data[key]["answer"]
                # 和头实体相似度大于尾实体
                if Levenshtein.ratio(ans, e1) > Levenshtein.ratio(ans, e2):
                    train_entity.append(e2)
                else:
                    train_entity.append(e1)

            # 求question/answer/entity 的数量
            train_triplet_num += len(train_question)
            test_triplet_num += len(test_question)

            # overlap of quetsion/ans/entity
            q_and = [val for val in train_question if val in test_question]
            e_and = [val for val in train_entity if val in test_entity]
            a_and = [val for val in train_answer if val in test_answer]
            i_and = [val for val in train_image if val in test_image]

            and_answer_num += len(a_and)
            and_entity_num += len(q_and)
            and_question_num += len(e_and)
            and_image_num += len(i_and)

            and_answer_class += len(set(a_and))
            and_entity_class += len(set(q_and))
            and_question_class += len(set(e_and))
            and_image_class += len(set(i_and))

            train_question_class += len(set(train_question))
            test_question_class += len(set(test_question))
            train_answer_class += len(set(train_answer))
            test_answer_class += len(set(test_answer))
            train_entity_class += len(set(train_entity))
            test_entity_class += len(set(test_entity))
            train_image_class += len(set(train_image))
            test_image_class += len(set(test_image))

        train_triplet_num = train_triplet_num / 5.
        test_triplet_num = test_triplet_num / 5.
        and_answer_num = and_answer_num / 5.
        and_entity_num = and_entity_num / 5.
        and_question_num = and_question_num / 5.
        and_image_num = and_image_num / 5.
        and_answer_class = and_answer_class / 5.
        and_entity_class = and_entity_class / 5.
        and_question_class = and_question_class / 5.
        and_image_class = and_image_class / 5.
        train_question_class = train_question_class / 5.
        test_question_class = test_question_class / 5.
        train_answer_class = train_answer_class / 5.
        test_answer_class = test_answer_class / 5.
        train_entity_class = train_entity_class / 5.
        test_entity_class = test_entity_class / 5.
        train_image_class = train_image_class / 5.
        test_image_class = test_image_class / 5.

        print(name + "_train_triplet_num:", train_triplet_num)
        print(name + "_test_triplet_num:", test_triplet_num)
        print(name + "_and_answer_num:", and_answer_num)
        print(name + "_and_entity_num:", and_entity_num)
        print(name + "_and_question_num:", and_question_num)
        print(name + "_and_image_num:", and_image_num)
        print(name + "_and_answer_class:", and_answer_class)
        print(name + "_and_entity_class:", and_entity_class)
        print(name + "_and_question_class:", and_question_class)
        print(name + "_and_image_class:", and_image_class)
        print(name + "_train_question_class:", train_question_class)
        print(name + "_test_question_class:", test_question_class)
        print(name + "_train_answer_class:", train_answer_class)
        print(name + "_test_answer_class:", test_answer_class)
        print(name + "_train_entity_class:", train_entity_class)
        print(name + "_test_entity_class:", test_entity_class)
        print(name + "_train_image_class:", train_image_class)
        print(name + "_test_image_class:", test_image_class)

    def data_analysis_zsl_and_general(self):
        # self.data_analysis("zsl")
        self.data_analysis("general")


if __name__ == '__main__':
    cfg = cfg()
    args = cfg.get_args()
    cfg.update_train_configs(args)
    runner = Runner(cfg)

    runner.get_new_json()
    # runner.get_new_all_json()

    runner.get_entity_filter()
    runner.get_all_entity()
    runner.fusion_answer_and_entity()

    # 此时得到的文件：all_qs_dict_release_combine_filter.json 是过滤好了的。
    # 包含三元组 5826

    # ans_set len: 833
    # entity_set len: 3294
    # ans_or_entity len: 3294
    # ans_and_entity len: 833
    # name = "all_qs_dict_release_combine_filter_fusion.json"

    # ans_set len: 500
    # entity_set len: 3027
    # ans_or_entity len: 3027
    # ans_and_entity len: 500
    # relation len: 108

    name = "all_qs_dict_release_combine_filter_fusion_500.json"

    runner.filter_top500_IQA_pair()
    runner.statistics_of_ans_and_entity(name=name)

    runner.filter_top500_IQA_pair()
    runner.deal_relation()
    runner.split_data()

    runner.preprocess_relation()
    runner.preprocess_fact()

    # runner.split_unseen_data()

    runner.get_fact_relation_matrix()

    runner.preprocess_json_in_order()

    # runner.data_analysis_zsl_and_general()
