
<p align="center">
    <a href="https://github.com/China-UK-ZSL/ZS-F-VQA"> <img src="https://raw.githubusercontent.com/zjunlp/openue/master/docs/images/logo_zju_klab.png" width="400"/></a>
</p>
<!-- [**中文**](https://github.com/zjukg/ZS-F-VQA/into/README_CN.md) | [**English**](https://github.com/zjukg/ZS-F-VQA/) -->



<!-- <p align="center">
    <font size=17><strong>Zero-shot VQA algorithm</strong>
    <center><strong>using Knowledge Graph</strong></center></font>
</p> -->

# Zero-shot VQA algorithm using Knowledge Graph


<!-- # ZS-F-VQA -->
[![](https://img.shields.io/badge/version-1.0.1-blue)](https://github.com/China-UK-ZSL/ZS-F-VQA)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arXiv-2107.05348-red)](http://arxiv.org/abs/2107.05348)

This repository is the official introduction of [*Zero-shot Visual Question Answering using Knowledge Graph*](https://arxiv.org/abs/2107.05348). This paper has been accepted by ISWC 2021 main conference.

```
Zhuo Chen, Jiaoyan Chen, Yuxia Geng, Jeff Z. Pan, Zonggang Yuan and Huajun Chen. Zero-shot Visual Question Answering using Knowledge Graph. ISWC2021 (International Semantic Web Conference) (CCF B).
```

# Author

[Zhuo Chen](https://github.com/hackerchenzhuo), [Jiaoyan Chen](https://github.com/ChenJiaoyan), [Yuxia Geng](https://github.com/genggengcss), Jeff Z. Pan, Zonggang Yuan and Huajun Chen


# Paper Introduction

## Abstract

In this work,  we propose a Zero-shot VQA algorithm using knowledge graph and a mask-based learning mechanism for better incorporating external knowledge, and present new answer-based Zero-shot VQA splits for the F-VQA dataset.


## Model

![Model_architecture](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/figures/Model_architecture.png)


## Experiments

- Explainable

![explainable](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/figures/all_explainable.png)

- Results

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

# How to Cite

If you use or extend our work, please cite the following paper:

```bigquery
@article{chen2021zero,
  title={Zero-shot Visual Question Answering using Knowledge Graph},
  author={Chen, Zhuo and Chen, Jiaoyan and Geng, Yuxia and Pan, Jeff Z and Yuan, Zonggang and Chen, Huajun},
  journal={arXiv preprint arXiv:2107.05348},
  year={2021}
}
```

