[**中文**](https://github.com/zjukg/ZS-F-VQA/into/README_CN.md) | [**English**](https://github.com/zjukg/ZS-F-VQA/)


<!-- <p align="center">
    <a href="https://github.com/zjukg/ZS-F-VQA"> <img src="https://raw.githubusercontent.com/zjunlp/openue/master/docs/images/logo_zju_klab.png" width="400"/></a>
</p>

<p align="center">
    <strong>使用知识图谱进行零样本视觉问答</strong>
</p> -->
<p align="center">
  	<font size=6><strong>使用知识图谱进行零样本视觉问答</strong></font>
</p>

<!-- # ZS-F-VQA -->
[![](https://img.shields.io/badge/version-1.0.1-blue)](https://github.com/China-UK-ZSL/ZS-F-VQA)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arXiv-2107.05348-red)](http://arxiv.org/abs/2107.05348)

这是针对我们论文[*Zero-shot Visual Question Answering using Knowledge Graph*](https://arxiv.org/abs/2107.05348) 的官方简介，目前论文已被 **ISWC 2021** 主会录用。
```
Zhuo Chen, Jiaoyan Chen, Yuxia Geng, Jeff Z. Pan, Zonggang Yuan and Huajun Chen. Zero-shot Visual Question Answering using Knowledge Graph. ISWC2021 (International Semantic Web Conference) (CCF B).
```

# 作者

[陈卓](https://github.com/hackerchenzhuo), [陈矫彦](https://github.com/ChenJiaoyan), [耿玉霞](https://github.com/genggengcss)、Jeff、苑宗港、陈华钧。


# 论文介绍

参考：[ISWC2021 | 当知识图谱遇上零样本视觉问答](https://mp.weixin.qq.com/s/mDWpgBLUbVZ7jju8oXSjEw)


## 摘要

>将外部知识引入视觉问答（Visual Question Answering, VQA）已成为一个重要的实际需求。现有的许多方法采用pipeline的模式，多模块分工进行跨模态知识处理和特征学习，但这种模式下，中间件的性能瓶颈会导致不可逆转的误差传播（Error Cascading）。此外，大多数已有工作都忽略了答案偏见问题——因为长尾效应的存在，真实世界许多答案在模型训练过程中可能不曾出现过（Unseen Answer）。

>在本文中，我们提出了一种适用于零样本视觉问答（ZS-VQA）的基于知识图谱的掩码机制，更好结合外部知识的同时，一定程度缓解了误差传播对于模型性能的影响。并在原有F-VQA数据集基础上，提供了基于Seen / Unseen答案类别为划分依据的零样本VQA数据集（ZS-F-VQA）。实验表明，我们的方法可以在该数据集下达到最佳性能，同时还可以显著增强端到端模型在标准F-VQA任务上的性能效果。


## 模型
方法包含两部分。

**第一部分**，我们提出三个特征空间以处理不同分布的信息：实体空间（Object Space）、语义空间（Semantic Space）、知识空间（Knowledge Space）的概念。其中：
- 实体空间主要处理图像/文本中存在的重点实体与知识库中存在实例的对齐；
- 语义空间关注视觉/语言的交互模态中蕴含的语义信息，其目的是让知识库中对应关系的表示在独立空间中进行特征逼近。
- 知识空间让 (问题，图像)组成的pair与答案直接对齐，建模的是间接知识，旨在挖掘多模态融合向量中存在的（潜层）知识。

**第二部分**是基于知识的答案掩码。

掩码技术技术广泛应用于预训练语言模型（PLM），其在训练阶段遮掩输入的片段，以自监督的方式学习语法语义。与这种方式不同，我们在输出阶段进行答案遮掩：给定输入图像/文本信息得到融合向量后，基于第一部分独立映射的特征空间和给定的超参数Ke / Kr，根据空间距离相似度在实体/语义空间中得到关于实体/关系的映射集，结合知识库三元组信息匹配得到答案候选集。答案候选集作为掩码的依据，在知识空间搜索得到的模糊答案的基础上进行掩码处理，最后进行答案排序。

此处的掩码类型的分为两种：硬掩码（hard mask）和软掩码（soft mask），主要作用于答案的判定分数（score），区别在于遮掩分数的多少。其作用场景分别为零样本场景和普通场景。零样本背景下领域偏移问题严重，硬掩码约束某种意义上对于答案命中效果的提升远大于丢失正确答案所带来的误差。而普通场景下过高的约束则容易导致较多的信息丢失，收益小于损失。
具体实验和讨论见原文。

![Model_architecture](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/figures/Model_architecture.png)



## 可解释性实验

![explainable](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/figures/all_explainable.png)



## 部分结果

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

更多细节见中文博客: [ISWC2021 | 当知识图谱遇上零样本视觉问答](https://mp.weixin.qq.com/s/mDWpgBLUbVZ7jju8oXSjEw)

# 如何引用

如果您使用或扩展我们的工作，请引用以下文章：

```bigquery
@article{chen2021zero,
  title={Zero-shot Visual Question Answering using Knowledge Graph},
  author={Chen, Zhuo and Chen, Jiaoyan and Geng, Yuxia and Pan, Jeff Z and Yuan, Zonggang and Chen, Huajun},
  journal={arXiv preprint arXiv:2107.05348},
  year={2021}
}
```


