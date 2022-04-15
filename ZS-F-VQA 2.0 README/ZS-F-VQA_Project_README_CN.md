[**中文**](https://github.com/zjukg/ZS-F-VQA/into/README_CN.md) | [**English**](https://github.com/zjukg/ZS-F-VQA/)

<!-- <p align="center">
    <a href="https://github.com/zjukg/ZS-F-VQA"> <img src="https://raw.githubusercontent.com/zjunlp/openue/master/docs/images/logo_zju_klab.png" width="400"/></a>
</p> -->

<p align="center">
    <font size=7><strong>使用知识图谱</strong>
    <center><strong>进行零样本视觉问答</strong></center></font>
</p>


<!-- # ZS-F-VQA -->
[![](https://img.shields.io/badge/version-1.0.1-blue)](https://github.com/China-UK-ZSL/ZS-F-VQA)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arXiv-2107.05348-red)](http://arxiv.org/abs/2107.05348)

这是针对我们项目[ZS-F-VQA](https://github.com/China-UK-ZSL/ZS-F-VQA)的官方实现代码。
这个模型是在**[Zero-shot Visual Question Answering using Knowledge Graph](https://arxiv.org/abs/2107.05348)**论文中提出来的，该论文已被**ISWC 2021**主会录用。

参考博客：
[ISWC2021 | 当知识图谱遇上零样本视觉问答](https://mp.weixin.qq.com/s/mDWpgBLUbVZ7jju8oXSjEw)

![Model_architecture](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/figures/Model_architecture.png)

# 项目成员
[陈卓](https://github.com/hackerchenzhuo), [陈矫彦](https://github.com/ChenJiaoyan), [耿玉霞](https://github.com/genggengcss)、Jeff、苑宗港、陈华钧。


# 环境要求

需要按以下命令去配置项目运行环境：
- `python >= 3.5`
- `PyTorch >= 1.6.0`

```bash
 pip install -r requirements.txt
```

## 数据

 5 **F-VQA** train / test data split 路径:
- ```data/KG_VQA/fvqa/exp_data/train_data```
- ```data/KG_VQA/fvqa/exp_data/test_data```

5 **ZS-F-VQA** train / test data split 路径: 
- ```data/KG_VQA/fvqa/exp_data/train_seen_data```
- ```data/KG_VQA/fvqa/exp_data/test_unseen_data```

答案路径： ``data/KG_VQA/data/FVQA/new_dataset_release/.``

**图像数据:**
- 图像文件夹 (put all your `.JPEG`/`.jpg` file here):
```data/KG_VQA/fvqa/exp_data/images/images```
- 图像特征: `fvqa-resnet-14x14.h5` 已预训练: [GoogleDrive](https://drive.google.com/file/d/1YG9hByw01_ZQ6_mKwehYiddG3x2Cxatu/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1ks84AWSXxJJ_7LwnzWdEnQ) (提取码:16vd)
- 原始图像： [FVQA](https://github.com/wangpengnorman/FVQA) with [download_link](https://www.dropbox.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip?dl=0).
- 其他 vqa 数据集: 生成预训练图像特征 ([Guidance](https://github.com/hexiang-hu/answer_embedding/issues/3) / [code](https://github.com/Cyanogenoid/pytorch-vqa/blob/master/preprocess-images.py))
- 生成的 `.h` 文件需要被放置于 :
```data/KG_VQA/fvqa/exp_data/common_data/.```
  
**答案 / 问题 词汇表:**
- 对应生成的文件 `answer.vocab.fvqa.json` & `question.vocab.fvqa.json`  已存在于 :
```data/KG_VQA/fvqa/exp_data/common_data/.```
- 其他 vqa 数据集: 参考代码 [process answer vocab](https://github.com/hexiang-hu/answer_embedding/blob/master/tools/preprocess_answer.py) 和 [process questions vocab](https://github.com/hexiang-hu/answer_embedding/blob/master/tools/preprocess_question.py)

---

### 预训练模型 ([url](https://www.dropbox.com/sh/vp5asuivqpiir5w/AAC3k_gELrP4ydNNok_o1vlYa?dl=0))

下载，并且覆盖： ```data/KG_VQA/fvqa/model_save```


## [模型参数](#content)
```
[--KGE {TransE,ComplEx,TransR,DistMult}] [--KGE_init KGE_INIT] [--GAE_init GAE_INIT] [--ZSL ZSL] [--entity_num {all,4302}] [--data_choice {0,1,2,3,4}]
               [--name NAME] [--no-tensorboard] --exp_name EXP_NAME [--dump_path DUMP_PATH] [--exp_id EXP_ID] [--random_seed RANDOM_SEED] [--freeze_w2v {0,1}]
               [--ans_net_lay {0,1,2}] [--fact_map {0,1}] [--relation_map {0,1}] [--now_test {0,1}] [--save_model {0,1}] [--joint_test_way {0,1}] [--top_rel TOP_REL]
               [--top_fact TOP_FACT] [--soft_score SOFT_SCORE] [--mrr MRR]
```

可训练模型选择: ```Up-Down```, `BAN`, `SAN`, `MLP`

**你可以把自定义的模型 (`.py`) 放置于 :** `main/code/model/.`

更多参数细节，见: ```code/config.py```


# 代码运行
```cd code```

**数据校验:**

- ```python deal_data.py --exp_name data_check```

**General VQA:**
- train:
```bash run_FVQA_train.sh```
- test:
```bash run_FVQA.sh```

**ZSL/GZSL VQA:**
- train:
```bash run_ZSL_train.sh```
- test:
```bash run_ZSL.sh```

**注意**: 
- 你可以打开 `.sh` 文件以进行 <a href="#Parameter">模型参数</a> 修改.

**结果存储:**
- Log file 存储于: ```code/dump```
- model 存储于: ```data/KG_VQA/fvqa/model_save```

<br />

## 可解释性

![explainable](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/figures/all_explainable.png)
# 结果

模型在 [ZS-F-VQA](https://github.com/China-UK-ZSL/ZS-F-VQA) 上达到了以下的性能：

| Model name                            | Top 1 Accuracy    | Top 3 Accuracy    | Top 10 Accuracy   | MR        |
| ------------------                    |  :-:              | :-:               | :-:               | :-:       |
| BAN†                                  |     13.14         |      26.92        | 46.90             | -         |
| SAN†                                  |     20.42         |      37.20        | 62.24             | 19.14     |
|                                       |                   | Our Method        |                   |           |
| k<sub>r</sub>=15, k<sub>e</sub>=3     |     **50.51**     |      70.44        | 84.24             | 9.27      |
| k<sub>r</sub>=15, k<sub>e</sub>=5     |     49.11         |      **71.17**    | 86.06             | 8.6       |
| k<sub>r</sub>=25, k<sub>e</sub>=15    |     40.21         |      67.04        | **88.51**         | 7.68      |
| k<sub>r</sub>=25, k<sub>e</sub>=25    |     35.87         |      61.86        | 88.09             | **7.3**   |

更多细节见: [link](https://paperswithcode.com/paper/zero-shot-visual-question-answering-using)

# 有关论文

如果您使用或拓展我们的工作，请引用以下论文：

```bigquery
@article{chen2021zero,
  title={Zero-shot Visual Question Answering using Knowledge Graph},
  author={Chen, Zhuo and Chen, Jiaoyan and Geng, Yuxia and Pan, Jeff Z and Yuan, Zonggang and Chen, Huajun},
  journal={arXiv preprint arXiv:2107.05348},
  year={2021}
}
```

感谢您对于本工作的关注：）

# 贡献

[MIT LICENSE](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/LICENSE)




