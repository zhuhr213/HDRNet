import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split as split
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader

from utils.MyNet1 import *
from utils.gen_bert_embedding import circRNABert
from utils.train_loop import validate
from utils.utils import myDataset, GradualWarmupScheduler, split_dataset
from utils.utils import read_csv


def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    seed = round(seed * random.random())
    print(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

fix_seed(1024)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_path = '/home/zhuhaoran/MyNet/'
filename = sys.argv[1] + '.tsv'



model__path = '/home/zhuhaoran/PrismNet-master/3-new-12w-0/'

tokenizer = BertTokenizer.from_pretrained(model__path, do_lower_case=False)
model = BertModel.from_pretrained(model__path)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model = model.eval()


sequences, structs, label = read_csv(base_path + 'clip_data/' + filename)

bert_embedding = circRNABert(list(sequences), model, tokenizer, device, 3)  # default k=3  # (N, 101, 768)
bert_embedding = bert_embedding.transpose([0, 2, 1])  # (N, 768, 101)

# num, cls = label0.shape
'''
label1 = label0.squeeze(1).astype(np.int32)
num_class = 2
label = np.zeros((len(label1), 2))
label[range(len(label1)), label1] = 1
'''
# label = F.one_hot(label0, 2)

structure = np.zeros((len(structs), 1, 101))  # (N, 1, 101)
for i in range(len(structs)):
    struct = structs[i].split(',')
    ti = [float(t) for t in struct]
    ti = np.array(ti).reshape(1, -1)
    structure[i] = np.concatenate([ti], axis=0)


# one_hot_embedding = convert_one_hot(sequences)  # [4, 101]

# bert_embedding = np.expand_dims(one_hot_embedding, axis=1)

[train_emb, train_struc, train_label], [test_emb, test_struc, test_label] = \
    split_dataset(bert_embedding, structure, label)  # , test_size=0.2, shuffle=True, stratify=label)

train_set = myDataset(train_emb, train_struc, train_label)
test_set = myDataset(test_emb, test_struc, test_label)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32 * 4, shuffle=False)

model_name = sys.argv[2]

model_file = '/home/zhuhaoran/MyNet/out/out1/' + model_name + '.tsv_best.pth'

model = MyNet10().to(device)
model.load_state_dict(torch.load(model_file))

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

met, y_all, p_all = validate(model, device, test_loader, criterion)

p = '/home/zhuhaoran/MyNet/cross_AUC_plot/HDRNet/'

save_path = p + filename.rstrip('.tsv')  # 保存的是被测试的
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.savetxt(save_path+'/y_true', y_all)
np.savetxt(save_path+'/y_pred', p_all)


print("> eval {} auc: {:.4f} acc: {:.4f}".format(filename, met.auc, met.acc))
# with open('/home/zhuhaoran/MyNet/outfile/cross_test_bertonly_results.txt', 'a') as f:
#     print(">eval {} >using {} auc: {:.4f} acc: {:.4f}".format(filename, model_name, met.auc, met.acc), file=f)
