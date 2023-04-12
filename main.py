import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from utils.HDRNet import HDRNet

from utils.gen_bert_embedding import circRNABert

import torch.utils.data
from transformers import BertModel, BertTokenizer

from utils.train_loop import train, validate
from utils.utils import read_csv, myDataset, GradualWarmupScheduler, param_num, split_dataset, seq2kmer

def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    # print("[Info] cudnn.deterministic set to True. CUDNN-optimized code may be slow.")

def main(args):

    try:
        from termcolor import cprint
    except ImportError:
        cprint = None

    try:
        from pycrayon import CrayonClient
    except ImportError:
        CrayonClient = None


    def log_print(text, color=None, on_color=None, attrs=None):
        if cprint is not None:
            cprint(text, color=color, on_color=on_color, attrs=attrs)
        else:
            print(text)


    fix_seed(args.seed)  # fix seed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_length = 101

    file_name = args.data_file
    data_path = args.data_path

    if args.train:
        sequences, structs, label = read_csv(os.path.join(data_path, file_name+'.tsv'))

        bert_model_path = args.BERT_model_path
        tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=False)
        model = BertModel.from_pretrained(bert_model_path)
        model = model.to(device)
        # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model = model.eval()
        bert_embedding = circRNABert(list(sequences), model, tokenizer, device, 3)  # (N, 99, 768)
        bert_embedding = bert_embedding.transpose([0, 2, 1])  # (N, 768, 99)

        structure = np.zeros((len(structs), 1, max_length))  # (N, 1, 101)
        for i in range(len(structs)):
            struct = structs[i].split(',')
            ti = [float(t) for t in struct]
            ti = np.array(ti).reshape(1, -1)
            structure[i] = np.concatenate([ti], axis=0)

        [train_emb, train_struc, train_label], [test_emb, test_struc, test_label] = \
            split_dataset(bert_embedding, structure, label)  # , test_size=0.2, shuffle=True, stratify=label)

        train_set = myDataset(train_emb, train_struc, train_label)
        test_set = myDataset(test_emb, test_struc, test_label)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=32 * 8, shuffle=False)

        model = HDRNet().to(device)
        # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=8, total_epoch=float(200), after_scheduler=None)

        best_auc = 0
        best_acc = 0
        best_epoch = 0

        model_save_path = args.model_save_path

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        early_stopping = args.early_stopping

        param_num(model)

        for epoch in range(1, 200):
            t_met = train(model, device, train_loader, criterion, optimizer, batch_size=32)
            v_met, _, _ = validate(model, device, test_loader, criterion)
            scheduler.step()
            lr = scheduler.get_lr()[0]
            color_best = 'green'
            if best_auc < v_met.auc:
                best_auc = v_met.auc
                best_acc = v_met.acc
                best_epoch = epoch
                color_best = 'red'
                path_name = os.path.join(model_save_path, file_name+'.pth')
                torch.save(model.state_dict(), path_name)
            if epoch - best_epoch > early_stopping:
                print("Early stop at %d, %s " % (epoch, 'HDRNet'))
                break
            line = '{} \t Train Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} lr: {:.6f}'.format(
                file_name, epoch, t_met.other[0], t_met.acc, t_met.auc, lr)
            log_print(line, color='green', attrs=['bold'])

            line = '{} \t Test  Epoch: {}     avg.loss: {:.4f} Acc: {:.2f}%, AUC: {:.4f} ({:.4f}) {}'.format(
                file_name, epoch, v_met.other[0], v_met.acc, v_met.auc, best_auc, best_epoch)
            log_print(line, color=color_best, attrs=['bold'])

        print("{} auc: {:.4f} acc: {:.4f}".format(file_name, best_auc, best_acc))

    if args.validate:  # validate only. WARNING: PLEASE FIX SEED BEFORE VALIDATION.
        sequences, structs, label = read_csv(os.path.join(data_path, file_name+'.tsv'))
        [train_seq, train_struc, train_label], [test_seq, test_struc, test_label] = \
            split_dataset(sequences, structs, label) 

        bert_model_path = args.BERT_model_path
        tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=False)
        model = BertModel.from_pretrained(bert_model_path)
        model = model.to(device)
        # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model = model.eval()
        bert_embedding = circRNABert(list(test_seq), model, tokenizer, device, 3)  # (N, 99, 768)
        test_emb = bert_embedding.transpose([0, 2, 1])  # (N, 768, 99)

        structure = np.zeros((len(test_struc), 1, max_length))  # (N, 1, 101)
        for i in range(len(test_struc)):
            struct = test_struc[i].split(',')
            ti = [float(t) for t in struct]
            ti = np.array(ti).reshape(1, -1)
            structure[i] = np.concatenate([ti], axis=0)

        test_set = myDataset(test_emb, structure, test_label)
        test_loader = DataLoader(test_set, batch_size=32 * 8, shuffle=False)

        model = HDRNet().to(device)
        model_file = os.path.join(args.model_save_path, file_name+'.pth')
        if not os.path.exists(model_file):
            print('Model file does not exitsts! Please train first and save the model')
            exit()
        model.load_state_dict(torch.load(model_file))

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

        met, y_all, p_all = validate(model, device, test_loader, criterion)
        best_auc = met.auc
        best_acc = met.acc
        print("{} auc: {:.4f} acc: {:.4f}".format(file_name, best_auc, best_acc))

    if args.dynamic_validate:   # perform dynamic prediction between K562 cell and HepG2 cell
        # cell_list = ['K562', 'HepG2']
        if file_name.endswith('K562'):
            model_file = file_name.replace('K562', 'HepG2')
        elif file_name.endswith('HepG2'):
            model_file = file_name.replace('HepG2', 'K562')
        else:
            print("Dynamic prediction only performs on K562 cells and HepG2 cells!")
            exit()
        sequences, structs, label = read_csv(os.path.join(data_path, file_name+'.tsv'))
        [train_seq, train_struc, train_label], [test_seq, test_struc, test_label] = \
            split_dataset(sequences, structs, label) 

        bert_model_path = args.BERT_model_path
        tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=False)
        model = BertModel.from_pretrained(bert_model_path)
        model = model.to(device)
        # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model = model.eval()
        bert_embedding = circRNABert(list(test_seq), model, tokenizer, device, 3)  # (N, 99, 768)
        test_emb = bert_embedding.transpose([0, 2, 1])  # (N, 768, 99)

        structure = np.zeros((len(test_struc), 1, max_length))  # (N, 1, 101)
        for i in range(len(test_struc)):
            struct = test_struc[i].split(',')
            ti = [float(t) for t in struct]
            ti = np.array(ti).reshape(1, -1)
            structure[i] = np.concatenate([ti], axis=0)

        test_set = myDataset(test_emb, structure, test_label)
        test_loader = DataLoader(test_set, batch_size=32 * 8, shuffle=False)

        model = HDRNet().to(device)
        # model_path = args.model_save_path
        model_path = os.path.join(args.model_save_path, model_file+'.pth')
        if not os.path.exists(model_path):
            print('The dynamic predition model {} does not exist! Please train first!'.format(model_file))
            exit()
        model = model.load_state_dict(torch.load(model_file))
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

        print('Using {} model to predict {} cell'.format(model_file, file_name))
        met, y_all, p_all = validate(model, device, test_loader, criterion)
        best_auc = met.auc
        best_acc = met.acc
        print("{} auc: {:.4f} acc: {:.4f}".format(file_name, best_auc, best_acc))

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Welcome to HDRNet!')
    parser.add_argument('--data_file', default='TIA1_Hela', type=str, help='RBP to train or validate')
    parser.add_argument('--data_path', default='./dataset', type=str, help='The data path')
    parser.add_argument('--BERT_model_path', default='./BERT_Model', type=str, help='BERT model path, in case you have another BERT')
    parser.add_argument('--model_save_path', default='./results/model', type=str, help='Save the trained model for dynamic prediction')

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--validate', default=False, action='store_true')
    parser.add_argument('--dynamic_validate', default=False, action='store_true')

    parser.add_argument('--seed', default=1024, type=int, help='The random seed')
    parser.add_argument('--early_stopping', type=int, default=20)

    args = parser.parse_args()
    main(args)
