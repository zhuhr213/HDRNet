import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    seq_length = len(seq)
    sub_seq = 'ATCG'
    import random
    rand1 = random.randint(0, 3)  # [0,3]
    rand2 = random.randint(0, 3)
    # seq = sub_seq[rand1] + seq + sub_seq[rand2]
    kmer = [seq[x:x + k] for x in range(seq_length - k + 1)]
    return kmer


def split_dataset(data1, data2, targets, valid_frac=0.2):
    ind0 = np.where(targets < 0.5)[0]
    ind1 = np.where(targets >= 0.5)[0]

    n_neg = int(len(ind0) * valid_frac)
    n_pos = int(len(ind1) * valid_frac)

    shuf_neg = np.random.permutation(len(ind0))
    shuf_pos = np.random.permutation(len(ind1))

    X_train1 = np.concatenate((data1[ind1[shuf_pos[n_pos:]]], data1[ind0[shuf_neg[n_neg:]]]))
    X_train2 = np.concatenate((data2[ind1[shuf_pos[n_pos:]]], data2[ind0[shuf_neg[n_neg:]]]))
    Y_train = np.concatenate((targets[ind1[shuf_pos[n_pos:]]], targets[ind0[shuf_neg[n_neg:]]]))
    train = [X_train1, X_train2, Y_train]

    X_test1 = np.concatenate((data1[ind1[shuf_pos[:n_pos]]], data1[ind0[shuf_neg[:n_neg]]]))
    X_test2 = np.concatenate((data2[ind1[shuf_pos[:n_pos]]], data2[ind0[shuf_neg[:n_neg]]]))
    Y_test = np.concatenate((targets[ind1[shuf_pos[:n_pos]]], targets[ind0[shuf_neg[:n_neg]]]))
    test = [X_test1, X_test2, Y_test]

    return train, test



def param_num(model):
    num_param0 = sum(p.numel() for p in model.parameters())
    num_param1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("===========================")
    print("Total params:", num_param0)
    print("Trainable params:", num_param1)
    print("Non-trainable params:", num_param0 - num_param1)
    print("===========================")


class myDataset(Dataset):
    def __init__(self, bert_embedding, structure, label):
        self.embedding = bert_embedding
        self.structs = structure
        self.label = label

    def __getitem__(self, index):
        embedding = self.embedding[index]
        struct = self.structs[index]
        label = self.label[index]

        return embedding, struct, label

    def __len__(self):
        return len(self.label)


def read_csv(path):
    # load sequences
    df = pd.read_csv(path, sep='\t', header=None)
    df = df.loc[df[0] != "Type"]

    Type = 0
    loc = 1
    Seq = 2
    Str = 3
    Score = 4
    label = 5

    rnac_set = df[Type].to_numpy()
    sequences = df[Seq].to_numpy()
    structs = df[Str].to_numpy()
    targets = df[label].to_numpy().astype(np.float32).reshape(-1, 1)
    return sequences, structs, targets


def read_csv_with_name(path):
    # load sequences
    df = pd.read_csv(path, sep='\t', header=None)
    df = df.loc[df[0] != "Type"]

    Type = 0
    loc = 1
    Seq = 2
    Str = 3
    Score = 4
    label = 5

    name = df[loc].to_numpy()
    sequences = df[Seq].to_numpy()
    structs = df[Str].to_numpy()
    targets = df[label].to_numpy().astype(np.float32).reshape(-1, 1)
    return name, sequences, structs, targets


def read_h5(file_path):
    f = h5py.File(file_path)
    embedding = np.array(f['bert_embedding']).astype(np.float32)
    structure = np.array(f['structure']).astype(np.float32)
    label = np.array(f['label']).astype(np.int32)
    f.close()
    return embedding, structure, label

def convert_one_hot(sequence, max_length=None):
    """convert DNA/RNA sequences to a one-hot representation"""
    one_hot_seq = []
    for seq in sequence:
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4,seq_length))
        index = [j for j in range(seq_length) if seq[j] == 'A']
        # print(index)
        one_hot[0,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'C']
        one_hot[1,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'G']
        one_hot[2,index] = 1
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        one_hot[3,index] = 1

        # handle boundary conditions with zero-padding
        if max_length:
            offset1 = int((max_length - seq_length)/2)
            offset2 = max_length - seq_length - offset1

            if offset1:
                one_hot = np.hstack([np.zeros((4,offset1)), one_hot])
            if offset2:
                one_hot = np.hstack([one_hot, np.zeros((4,offset2))])

        one_hot_seq.append(one_hot)

    # convert to numpy array
    one_hot_seq = np.array(one_hot_seq)

    return one_hot_seq


def convert_one_hot2(sequence, attention, max_length=None):
    """convert DNA/RNA sequences to a one-hot representation"""
    one_hot_seq = []
    for seq in sequence:
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4,seq_length))
        index = [j for j in range(seq_length) if seq[j] == 'A']
        for i in index:
            one_hot[0,i] = attention[i]
        index = [j for j in range(seq_length) if seq[j] == 'C']
        for i in index:
            one_hot[1,i] = attention[i]
        index = [j for j in range(seq_length) if seq[j] == 'G']
        for i in index:
            one_hot[2,i] = attention[i]
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        for i in index:
            one_hot[3,i] = attention[i]

        # handle boundary conditions with zero-padding
        if max_length:
            offset1 = int((max_length - seq_length)/2)
            offset2 = max_length - seq_length - offset1

            if offset1:
                one_hot = np.hstack([np.zeros((4,offset1)), one_hot])
            if offset2:
                one_hot = np.hstack([one_hot, np.zeros((4,offset2))])

        one_hot_seq.append(one_hot)

    # convert to numpy array
    one_hot_seq = np.array(one_hot_seq)

    return one_hot_seq

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
