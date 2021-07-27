# ZS-F-VQA

Code and Data for the paper: [*Zero-shot Visual Question Answering using Knowledge Graph*](https://arxiv.org/abs/2107.05348) [ ISWC 2021 ]. ![](https://img.shields.io/badge/version-1.0.0-blue)

>In this work,  we propose a Zero-shot VQA algorithm using knowledge graphs and a mask-based learning mechanism for better incorporating external knowledge, and present new answer-based Zero-shot VQA splits for the F-VQA dataset. 

## Model Architecture
![Model_architecture](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/figures/Model_architecture.png)

<br />

## Usage

### Requirements
- `python >= 3.5`
- `PyTorch >= 1.0.0`

For more detail of requirements: 
```bash
 pip install -r requirements.txt
```

### Data

Location of 5 **F-VQA** train / test data split:
- ```data/KG_VQA/fvqa/exp_data/train_data```
- ```data/KG_VQA/fvqa/exp_data/test_data```

Location of 5 **ZS-F-VQA** train / test data split: 
- ```data/KG_VQA/fvqa/exp_data/train_seen_data```
- ```data/KG_VQA/fvqa/exp_data/test_unseen_data```

Answers are available at ``data/KG_VQA/data/FVQA/new_dataset_release/.``

Image folder:
```data/KG_VQA/fvqa/exp_data/images/images```
Image feature:
- **fvqa-resnet-14x14.h5 pretrained**:  You could generate a pretrained image feature via this way ([Guidance](https://github.com/hexiang-hu/answer_embedding/issues/3))

The generated file ```fvqa-resnet-14x14.h5``` (about 2.3 GB) should be placed in : 

```data/KG_VQA/fvqa/exp_data/common_data/```

---

### Pretrained Model ([url](https://www.dropbox.com/sh/vp5asuivqpiir5w/AAC3k_gELrP4ydNNok_o1vlYa?dl=0))

Download it and overwrite ```data/KG_VQA/fvqa/model_save```


### [Parameter](#content)
```
[--KGE {TransE,ComplEx,TransR,DistMult}] [--KGE_init KGE_INIT] [--GAE_init GAE_INIT] [--ZSL ZSL] [--entity_num {all,4302}] [--data_choice {0,1,2,3,4}]
               [--name NAME] [--no-tensorboard] --exp_name EXP_NAME [--dump_path DUMP_PATH] [--exp_id EXP_ID] [--random_seed RANDOM_SEED] [--freeze_w2v {0,1}]
               [--ans_net_lay {0,1,2}] [--fact_map {0,1}] [--relation_map {0,1}] [--now_test {0,1}] [--save_model {0,1}] [--joint_test_way {0,1}] [--top_rel TOP_REL]
               [--top_fact TOP_FACT] [--soft_score SOFT_SCORE] [--mrr MRR]
```

Available model for training: ```Up-Down```, `BAN`, `SAN`, `MLP`

For more details: ```code/config.py```

---

### Running
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

**Result:**
- Log file will be saved to: ```code/dump```
- model will be saved to: ```data/KG_VQA/fvqa/model_save```

<br />

## Explainable

![explainable](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/figures/all_explainable.png)

<br />

## Acknowledgements
Thanks for the following released works:
>[SciencePlots](https://github.com/garrettj403/SciencePlots), [ramen](https://github.com/erobic/ramen), [GAE](https://github.com/zfjsail/gae-pytorch), [vqa-winner-cvprw-2017](https://github.com/markdtw/vqa-winner-cvprw-2017), [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch), [VQA](https://github.com/Shivanshu-Gupta/Visual-Question-Answering), [BAN](https://github.com/jnhwkim/ban-vqa), [commonsense-kg-completion](https://github.com/allenai/commonsense-kg-completion), [bottom-up-attention-vqa](https://github.com/hengyuan-hu/bottom-up-attention-vqa), [FVQA](https://github.com/wangpengnorman/FVQA), [answer_embedding](https://github.com/hexiang-hu/answer_embedding), [torchlight](https://github.com/RamonYeung/torchlight)


## Cite:
```bigquery
@article{chen2021zero,
  title={Zero-shot Visual Question Answering using Knowledge Graph},
  author={Chen, Zhuo and Chen, Jiaoyan and Geng, Yuxia and Pan, Jeff Z and Yuan, Zonggang and Chen, Huajun},
  journal={arXiv preprint arXiv:2107.05348},
  year={2021}
}
```


