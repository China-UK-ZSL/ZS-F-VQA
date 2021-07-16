import os
import json

import torch
import torch.nn as nn
import time
import pdb


class Metrics:
    """
    Stores accuracy (score), loss and timing info
    """

    def __init__(self, topnum=10):

        self.topnum = topnum
        self.total_loss = 0

        self.correct_1 = 0
        self.correct_3 = 0
        self.correct_10 = 0
        self.acc_all = 0
        self.acc_1 = 0
        self.acc_3 = 0
        self.acc_10 = 0
        self.num_examples = 0
        self.num_epoch = 0

        self.mrr = 0
        self.mr = 0
        self.mrr_all = 0
        self.mr_all = 0

    def update_per_batch(self, loss, answers, pred):
        self.total_loss += loss
        self.num_epoch += 1
        if self.topnum == 10:
            top1, top3, top10 = self.batch_accuracy_10(pred, answers)
        elif self.topnum == 50:
            top1, top3, top10 = self.batch_accuracy_50(pred, answers)
        elif self.topnum == 200:
            top1, top3, top10 = self.batch_accuracy_200(pred, answers)
        self.num_examples += top1.shape[0]

        self.correct_1 += top1.sum().item()
        self.correct_3 += top3.sum().item()
        self.correct_10 += top10.sum().item()

        #
        mrr_tmp, mr_tmp = self.batch_mr_mrr(pred, answers)
        self.mrr_all += mrr_tmp.sum().item()
        self.mr_all += mr_tmp.sum().item()

    def update_per_epoch(self):
        self.acc_1 = 100 * (self.correct_1 / self.num_examples)
        self.acc_3 = 100 * (self.correct_3 / self.num_examples)
        self.acc_10 = 100 * (self.correct_10 / self.num_examples)

        self.mr = self.mr_all / self.num_examples
        self.mrr = self.mrr_all / self.num_examples

        self.total_loss = self.total_loss / self.num_epoch
        self.acc_all = self.acc_1 + self.acc_3 + self.acc_10

    def batch_accuracy_10(self, predicted, true):
        """ Compute the accuracies for a batch of predictions and answers """
        # (Pdb) predicted.shape
        # torch.Size([128, 500])
        # (Pdb) true.shape
        # torch.Size([128, 500])
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
        return top1, top3, top10

    def batch_accuracy_50(self, predicted, true):
        """ Compute the accuracies for a batch of predictions and answers """
        if len(true.shape) == 3:
            true = true[0]
        _, ok = predicted.topk(50, dim=1)
        agreeing_all = torch.zeros([predicted.shape[0], 1], dtype=torch.float).cuda()
        for i in range(50):
            tmp = ok[:, i].reshape(-1, 1)
            agreeing_all += true.gather(dim=1, index=tmp)
            if i == 9:
                top10 = (agreeing_all * 0.3).clamp(max=1)
            if i == 29:
                top30 = (agreeing_all * 0.3).clamp(max=1)
            if i == 49:
                top50 = (agreeing_all * 0.3).clamp(max=1)

        return top10, top30, top50

    def batch_accuracy_200(self, predicted, true):
        """ Compute the accuracies for a batch of predictions and answers """
        if len(true.shape) == 3:
            true = true[0]
        _, ok = predicted.topk(200, dim=1)
        agreeing_all = torch.zeros([predicted.shape[0], 1], dtype=torch.float).cuda()
        for i in range(200):
            tmp = ok[:, i].reshape(-1, 1)
            agreeing_all += true.gather(dim=1, index=tmp)
            if i == 79:
                top10 = (agreeing_all * 0.3).clamp(max=1)
            if i == 149:
                top30 = (agreeing_all * 0.3).clamp(max=1)
            if i == 199:
                top50 = (agreeing_all * 0.3).clamp(max=1)

        return top10, top30, top50

    def batch_mr_mrr(self, predicted, true):
        if len(true.shape) == 3:
            true = true[0]

        # 计算
        top_rank = predicted.shape[1]
        batch_size = predicted.shape[0]
        _, predict_ans_rank = predicted.topk(top_rank, dim=1)  # 答案排名的坐标 batchsize * 500
        _, real_ans = true.topk(1, dim=1)  # 真正的答案：batchsize * 1

        # 扩充维度
        real_ans = real_ans.expand(batch_size, top_rank)
        ans_different = torch.abs(predict_ans_rank - real_ans)
        # 此时为0的位置就是预测正确的位置
        _, real_ans_list = ans_different.topk(top_rank, dim=1)  # 此时最后一位的数值就是正确答案在预测答案里面的位置,为 0
        real_ans_list = real_ans_list + 1.0
        mr = real_ans_list[:, -1].reshape(-1, 1).to(torch.float64)
        mrr = 1.0 / mr
        # pdb.set_trace()

        return mrr, mr

    # def print(self, epoch):
    #     print("Epoch {} Score {:.2f} Loss {}".format(epoch, 100 * self.raw_score / self.num_examples,
    #                                                  self.loss / self.num_examples))


# def accumulate_metrics(epoch, train_metrics, val_metrics, val_per_type_metric,
#                        best_val_score,
#                        best_val_epoch, save_val_metrics=True):
#     stats = {
#         "epoch": epoch,

#         "train_loss": float(train_metrics.loss),
#         "train_raw_score": float(train_metrics.raw_score),
#         "train_normalized_score": float(train_metrics.normalized_score),
#         "train_upper_bound": float(train_metrics.upper_bound),
#         "train_score": float(train_metrics.score),
#         "train_num_examples": train_metrics.num_examples,

#         "train_time": train_metrics.end_time - train_metrics.start_time,
#         "val_time": val_metrics.end_time - val_metrics.start_time
#     }
#     if save_val_metrics:
#         stats["val_raw_score"] = float(val_metrics.raw_score)
#         stats["val_normalized_score"] = float(val_metrics.normalized_score)
#         stats["val_upper_bound"] = float(val_metrics.upper_bound)
#         stats["val_loss"] = float(val_metrics.loss)
#         stats["val_score"] = float(val_metrics.score)
#         stats["val_num_examples"] = val_metrics.num_examples
#         stats["val_per_type_metric"] = val_per_type_metric.get_json()

#         stats["best_val_score"] = float(best_val_score)
#         stats["best_epoch"] = best_val_epoch

#     print(json.dumps(stats, indent=4))
#     return stats
