# Dynamic characterization and interpretation for protein–RNA interactions across diverse cellular conditions using HDRNet

<p align="left">
  <a href="https://github.com/zhuhr213/HDRNet">
    <img src="https://img.shields.io/badge/HDRNet-python-orange">
  </a>
  <a href="https://github.com/zhuhr213/HDRNet/stargazers">
    <img src="https://img.shields.io/github/stars/zhuhr213/HDRNet">
  </a>
  <a href="https://github.com/zhuhr213/HDRNet/network/members">
    <img src="https://img.shields.io/github/forks/zhuhr213/HDRNet">
  </a>
  <a href="https://github.com/zhuhr213/HDRNet/issues">
    <img src="https://img.shields.io/github/issues/zhuhr213/HDRNet">
  </a>
  <a href="https://github.com/zhuhr213/HDRNet/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/zhuhr213/HDRNet">
  </a>
</p>

`HDRNet` is a python package developed for RNA-RBP interaction sites identification and high-attention binding peak recognization using CNN based deep network.

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#Usage)
- [Data Availability](#data-availability)
- [License](#license)

# Overview
RNA-binding proteins (RBPs) play crucial roles in various genetic contexts, and understanding the interactions between RNAs and RBPs under distinct conditions forms the basis for comprehending RNA functions and post-transcriptional regulatory mechanisms. However, the limitations of current computational methods in accounting for the diversity of cellular conditions have posed a formidable challenge for the cross-prediction of RNA-protein binding events across different cell lines. Here, we developed HDRNet, an end-to-end deep learning-based framework to precisely predict dynamic RBP binding events. To enrich the available information of RNA sequences, multi-source information including in vivo RNA secondary structure information and bio-language features are integrated to characterize both the sequence and structural features of RNA. After that, hierarchical multi-scale residual network (HMRN) are leveraged to comprehend the contextual dependencies between the nucleotides and their structure, while a deep protein-RNA binding predictor (DPRBP) is proposed to learn the underlying representation and selects the crucial nucleotide tokens in a synergistic manner. We demonstrate the effectiveness of HDRNet by comparing it to state-of-the-art RNA-binding event identification methods on 261 linear RNA datasets from eCLIP data under both static and dynamic cellular conditions. Our results indicate that the proposed method outperforms the comparative approaches and is particularly suitable for dynamic predictions. In addition, we conducted motif and interpretation analysis to provide fresh insights into the pathological mechanisms underlying RNA-RBP from various perspectives. Our functional genomic analysis further explores the gene-human disease association, uncovering previously uncharacterized observations across a broad range of genetic disorders. 
![HDRNet](https://github.com/zhuhr213/HDRNet/blob/master/HDRNet.png)  

## NOTICE

Due to the capacity limiation of Github, we put the relevant files (including the BERT model and all datasets) in our webserver <a href="http://39.104.69.176:5050/">webserver</a> and <a href="https://figshare.com/articles/software/HDRNet\_zip/21454713">figshare</a>. All source code, data and model are open source and can be downloaded from GitHub or any other processes.


# System Requirements
## Hardware requirements
`HDRNet` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for *Linux*. The package has been tested on the following systems:
+ Linux: Ubuntu 20.04

### Python Dependencies
`HDRNet` mainly depends on the Python scientific stack.
```
numpy
scipy
pytorch
scikit-learn
scikit-image
pandas
transformers
shap
```
For specific setting, please see <a href="https://github.com/zhuhr213/HDRNet/blob/master/requirements.txt">requirements</a> or <a href="https://github.com/zhuhr213/HDRNet/blob/master/HDRNet.yml">yml</a>.

# Installation Guide:

### We recommend using a conda environment to build HDRNet.

```
$ conda env create -f HDRNet.yml 
```

# Usage

The BERT model and all to be trained datasets should first be download and put into the corresponding

folder. Then, you can train a model with a certain RBP dataset using the following command:

```python
python main.py --data_file TIA1_Hela --train --BERT_model_path ./BERT_Model --model_save_path ./results/model
```

The --BERT_model_path parameter can be any BERT model path that takes RNA sequences as input,  
and can be identified by the transformers module in python. All main parameters are listed in main.py.

After training, you can validate the model by using :

```python
python main.py --data_file TIA1_Hela --validate --BERT_model_path ./BERT_Model
```

Take TIA1_Hela dataset as an example, the validation is upon the trained TIA1_Hela model stored  
in the --model_save_path.

We also provide dynamic prediction tasks by using:

```python
python main.py --data_file some_dataset --dynamic_validate --BERT_model_path ./BERT_Model
```

The --data_file is the dataset to be validated. For example, if the input data file is AARS_K562,  
then the AARS_HepG2 model will be loaded to valid the AARS_K562 dataset. The trained model should  
be saved first.

The prediction results will be displayed automatically. If you need to save the results, please  
specify the path yourself.

We also provide users with a complete prediction process in the <a href="https://github.com/zhuhr213/HDRNet/blob/master/high_attention_region_recognization.ipynb">tutorial</a>, where the prediction tasks and the high-attention binding region visualization are included.
![HDRNet](https://github.com/zhuhr213/HDRNet/blob/master/results/high_attention_region_plot/out.png) 


# Data Availability
We present a user-friendly web server for the HDRNet method at <a href="http://39.104.69.176:5050/">webserver</a>, which enables users to determine whether a given RNA sequence is a binding site for RNA-binding proteins (RBPs). Moreover, all supporting source code and data can be downloaded from <a href="https://github.com/zhuhr213/HDRNet">here</a> and <a href="https://figshare.com/articles/software/HDRNet\_zip/21454713">figshare</a>.
                                    

# License
This project is covered under the **MIT License**.


Thank you for using HDRNet! Any questions, suggestions or advice are welcome!  
Contact: lixt314@jlu.edu.cn, zhuhr20@mails.jlu.edu.cn
