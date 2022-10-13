# HDRNet: high-throughput dynamic cellular RNA-binding event identification in diverse cellular conditions

HDRNet is developed for RNA-RBP interaction sites identification using CNN based deep network.

![HDRNet](https://www.baidu.com/img/bd_logo1.png)  

## NOTICE

Due to the capacity limiation of Github, we put the relevant files (including the BERT model and all datasets) in our webserver  ******. All source code, data and model are open source and can be downloaded from GitHub or any other processes.

## Requirements

* Python 3.9
* Pytorch 1.10 with CUDA 11.3
* scipy 1.9.1
* sklearn 1.0.2
* pandas 1.4.1

NOTE: The *transformers* package is already implemented in *src* folder.

Any missing packages, please use pip install or conda install to install.

## Usage

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

Thank you for using HDRNet! Any questions, suggestions or advice are welcome!  
Contact: zhuhr20@mails.jlu.edu.cn
