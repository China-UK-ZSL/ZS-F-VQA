<!-- [**中文**](https://github.com/zjukg/ZS-F-VQA/into/README_CN.md) | [**English**](https://github.com/zjukg/ZS-F-VQA/) -->


<p align="center">
    <a href="https://github.com/China-UK-ZSL/ZS-F-VQA"> <img src="https://raw.githubusercontent.com/zjunlp/openue/master/docs/images/logo_zju_klab.png" width="400"/></a>
</p>



<!-- <p align="center">
    <font size=17><strong>Zero-shot VQA algorithm using Knowledge Graph</strong></font>
</p> -->

# Zero-shot VQA algorithm using Knowledge Graph
<!-- # ZS-F-VQA -->
[![](https://img.shields.io/badge/version-1.0.1-blue)](https://github.com/China-UK-ZSL/ZS-F-VQA)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arXiv-2107.05348-red)](http://arxiv.org/abs/2107.05348)

This repository is the official implementation of [ZS-F-VQA](https://github.com/China-UK-ZSL/ZS-F-VQA), which is model proposed in a paper: 
[*Zero-shot Visual Question Answering using Knowledge Graph*](https://arxiv.org/abs/2107.05348), accepted by **ISWC 2021** main conference. 


# Contributors
[Zhuo Chen](https://github.com/hackerchenzhuo), [Jiaoyan Chen](https://github.com/ChenJiaoyan), [Yuxia Geng](https://github.com/genggengcss), Jeff Z. Pan, Zonggang Yuan and Huajun Chen
# Brief Introduction
>In this work,  we propose a Zero-shot VQA algorithm using knowledge graph and a mask-based learning mechanism for better incorporating external knowledge, and present new answer-based Zero-shot VQA splits for the F-VQA dataset.

![Model_architecture](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/figures/Model_architecture.png)



# Requirements

- `python >= 3.5`
- `PyTorch >= 1.6.0`

For more detail of requirements: 
```bash
 pip install -r requirements.txt
```

## Data

Location of 5 **F-VQA** train / test data split:
- ```data/KG_VQA/fvqa/exp_data/train_data```
- ```data/KG_VQA/fvqa/exp_data/test_data```

Location of 5 **ZS-F-VQA** train / test data split: 
- ```data/KG_VQA/fvqa/exp_data/train_seen_data```
- ```data/KG_VQA/fvqa/exp_data/test_unseen_data```

Answers are available at ``data/KG_VQA/data/FVQA/new_dataset_release/.``

**Image:**
- Image folder (put all your `.JPEG`/`.jpg` file here):
```data/KG_VQA/fvqa/exp_data/images/images```
- Image feature: `fvqa-resnet-14x14.h5` pretrained: [GoogleDrive](https://drive.google.com/file/d/1YG9hByw01_ZQ6_mKwehYiddG3x2Cxatu/view?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/1ks84AWSXxJJ_7LwnzWdEnQ) (password:16vd)
- Origin images are available at [FVQA](https://github.com/wangpengnorman/FVQA) with [download_link](https://www.dropbox.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip?dl=0).
- Other vqa dataset: you could generate a pretrained image feature via this way ([Guidance](https://github.com/hexiang-hu/answer_embedding/issues/3) / [code](https://github.com/Cyanogenoid/pytorch-vqa/blob/master/preprocess-images.py))
- The generated `.h` file should be placed in :
```data/KG_VQA/fvqa/exp_data/common_data/.```
  
**Answer / Qusetion vocab:**
- The generated file `answer.vocab.fvqa.json` & `question.vocab.fvqa.json`  now are available at :
```data/KG_VQA/fvqa/exp_data/common_data/.```
- Other vqa dataset: code for [process answer vocab](https://github.com/hexiang-hu/answer_embedding/blob/master/tools/preprocess_answer.py) and [process questions vocab](https://github.com/hexiang-hu/answer_embedding/blob/master/tools/preprocess_question.py)

---

## Pretrained Model ([url](https://www.dropbox.com/sh/vp5asuivqpiir5w/AAC3k_gELrP4ydNNok_o1vlYa?dl=0))

Download it and overwrite ```data/KG_VQA/fvqa/model_save```


## [Parameter](#content)
```
[--KGE {TransE,ComplEx,TransR,DistMult}] [--KGE_init KGE_INIT] [--GAE_init GAE_INIT] [--ZSL ZSL] [--entity_num {all,4302}] [--data_choice {0,1,2,3,4}]
               [--name NAME] [--no-tensorboard] --exp_name EXP_NAME [--dump_path DUMP_PATH] [--exp_id EXP_ID] [--random_seed RANDOM_SEED] [--freeze_w2v {0,1}]
               [--ans_net_lay {0,1,2}] [--fact_map {0,1}] [--relation_map {0,1}] [--now_test {0,1}] [--save_model {0,1}] [--joint_test_way {0,1}] [--top_rel TOP_REL]
               [--top_fact TOP_FACT] [--soft_score SOFT_SCORE] [--mrr MRR]
```

Available model for training: ```Up-Down```, `BAN`, `SAN`, `MLP`

**You can try your own model via adding it (`.py`) to :** `main/code/model/.`

For more details: ```code/config.py```

---

## Running
```cd code```

**For data check:**

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

**Note**: 
- you can open the `.sh` file for <a href="#Parameter">parameter</a> modification.

**Result path:**
- Log file will be saved to: ```code/dump```
- model will be saved to: ```data/KG_VQA/fvqa/model_save```

<br />

## Explainable

![explainable](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/figures/all_explainable.png)

<br />

## Acknowledgements
Thanks for the following released works:
>[SciencePlots](https://github.com/garrettj403/SciencePlots), [ramen](https://github.com/erobic/ramen), [GAE](https://github.com/zfjsail/gae-pytorch), [vqa-winner-cvprw-2017](https://github.com/markdtw/vqa-winner-cvprw-2017), [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch), [VQA](https://github.com/Shivanshu-Gupta/Visual-Question-Answering), [BAN](https://github.com/jnhwkim/ban-vqa), [commonsense-kg-completion](https://github.com/allenai/commonsense-kg-completion), [bottom-up-attention-vqa](https://github.com/hengyuan-hu/bottom-up-attention-vqa), [FVQA](https://github.com/wangpengnorman/FVQA), [answer_embedding](https://github.com/hexiang-hu/answer_embedding), [torchlight](https://github.com/RamonYeung/torchlight)

# Results

ZSL Performance(%) on : [ZS-F-VQA](https://github.com/China-UK-ZSL/ZS-F-VQA)

| Model name                            | Top 1 Accuracy    | Top 3 Accuracy    | Top 10 Accuracy   | MR        |
| ------------------                    |  :-:              | :-:               | :-:               | :-:       |
| BAN†                                  |     13.14         |      26.92        | 46.90             | -         |
| SAN†                                  |     20.42         |      37.20        | 62.24             | 19.14     |
|                                       |                   | Our Method        |                   |           |
| k<sub>r</sub>=15, k<sub>e</sub>=3     |     **50.51**     |      70.44        | 84.24             | 9.27      |
| k<sub>r</sub>=15, k<sub>e</sub>=5     |     49.11         |      **71.17**    | 86.06             | 8.6       |
| k<sub>r</sub>=25, k<sub>e</sub>=15    |     40.21         |      67.04        | **88.51**         | 7.68      |
| k<sub>r</sub>=25, k<sub>e</sub>=25    |     35.87         |      61.86        | 88.09             | **7.3**   |

For more details: [link](https://paperswithcode.com/paper/zero-shot-visual-question-answering-using)


# Papers for the Project & How to Cite

If you use or extend our work, please cite the following paper:

```bigquery
@article{chen2021zero,
  title={Zero-shot Visual Question Answering using Knowledge Graph},
  author={Chen, Zhuo and Chen, Jiaoyan and Geng, Yuxia and Pan, Jeff Z and Yuan, Zonggang and Chen, Huajun},
  journal={arXiv preprint arXiv:2107.05348},
  year={2021}
}
```


# Contributing

[MIT LICENSE](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/LICENSE)
