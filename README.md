# ZS-F-VQA

Code and Data for the paper: "Zero-shot Visual Question Answering using Knowledge Graphs".

In this work,  we propose a Zero-shot VQA algorithm using knowledge graphs and a mask-based learning mechanism for better incorporating external knowledge, and present new answer-based Zero-shot VQA splits for the F-VQA dataset. 


### Requirements
- `python >= 3.5`
- `PyTorch >= 1.0.0`
For more detail of requirements :
	$ pip install -r requirements.txt


### Data
"FVQA: Fact-based Visual Question Answering": [GitHub link](https://github.com/wangpengnorman/FVQA)

Which contains the url to F-VQA dataset: 
 - [Dataset_release](https://www.dropbox.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip?dl=0)
 - [Images](https://www.dropbox.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip?dl=0&file_subpath=%2Fnew_dataset_release%2Fimages)

for all images and 5 original train/test data splits

fvqa-resnet-14x14.h5 pretrained: 
 - Via [here](https://github.com/hexiang-hu/answer_embedding) 

### Model




### Running the code
General VQA:
```bash
Bash run_General_VQA.sh
```

ZSL/GZSL VQA:
```bash
Bash run_Zsl_VQA.sh
```

Results will be saved to ./code/KG_VQA/dump



### Acknowledgements
Thanks for their previous work:
>[SciencePlots](https://github.com/garrettj403/SciencePlots), [ramen](https://github.com/erobic/ramen), [GAE](https://github.com/zfjsail/gae-pytorch), [vqa-winner-cvprw-2017](https://github.com/markdtw/vqa-winner-cvprw-2017), [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch), [VQA](https://github.com/Shivanshu-Gupta/Visual-Question-Answering), [BAN](https://github.com/jnhwkim/ban-vqa), [commonsense-kg-completion](https://github.com/allenai/commonsense-kg-completion), [bottom-up-attention-vqa](https://github.com/hengyuan-hu/bottom-up-attention-vqa), [FVQA](https://github.com/wangpengnorman/FVQA), [answer_embedding](https://github.com/hexiang-hu/answer_embedding)


