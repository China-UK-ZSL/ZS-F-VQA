# ZS-F-VQA

Code and Data for the paper: [*Zero-shot Visual Question Answering using Knowledge Graphs*](https://arxiv.org/abs/2107.05348).

>In this work,  we propose a Zero-shot VQA algorithm using knowledge graphs and a mask-based learning mechanism for better incorporating external knowledge, and present new answer-based Zero-shot VQA splits for the F-VQA dataset. 

### Model
![Model_architecture](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/figures/Model_architecture.png)


### Requirements
- `python >= 3.5`
- `PyTorch >= 1.0.0`

For more detail of requirements: 
```bash
 pip install -r requirements.txt
```

 


### Data
[*FVQA: Fact-based Visual Question Answering*](chrome-extension://cdonnmffkdaoajfknoeeecmchibpmkmg/assets/pdf/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1606.05433.pdf)**:** [GitHub link](https://github.com/wangpengnorman/FVQA)

 F-VQA dataset: 
 - [Dataset_release](https://www.dropbox.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip?dl=0)
 - [Images](https://www.dropbox.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip?dl=0&file_subpath=%2Fnew_dataset_release%2Fimages)

Location of 5 original train/test data split: to be done

Location of 5 new **ZSL** train/test data split: to be done

Image feature:
- fvqa-resnet-14x14.h5 pretrained:  [here](https://github.com/hexiang-hu/answer_embedding) 



### Running
General VQA:
```bash
Bash run_General_VQA.sh
```

ZSL/GZSL VQA:
```bash
Bash run_Zsl_VQA.sh
```

Results will be saved to: to be done

### Explainable
![explainable](https://github.com/China-UK-ZSL/ZS-F-VQA/blob/main/figures/all_explainable.png)

### Acknowledgements
Thanks for the following works:
>[SciencePlots](https://github.com/garrettj403/SciencePlots), [ramen](https://github.com/erobic/ramen), [GAE](https://github.com/zfjsail/gae-pytorch), [vqa-winner-cvprw-2017](https://github.com/markdtw/vqa-winner-cvprw-2017), [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch), [VQA](https://github.com/Shivanshu-Gupta/Visual-Question-Answering), [BAN](https://github.com/jnhwkim/ban-vqa), [commonsense-kg-completion](https://github.com/allenai/commonsense-kg-completion), [bottom-up-attention-vqa](https://github.com/hengyuan-hu/bottom-up-attention-vqa), [FVQA](https://github.com/wangpengnorman/FVQA), [answer_embedding](https://github.com/hexiang-hu/answer_embedding)


### TODO:
**to be released:**
- [ ] code 
- [ ] pretrain model 
- [ ] running script 
- [ ] finish readme
- [ ] more dataset
- [ ] more detailed result


