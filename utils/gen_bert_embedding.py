import numpy as np
import torch
import torch.utils.data
from utils.utils import read_csv


def seq2kmer_bert(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    seq_length = len(seq)
    # sub_seq = 'ATCG'
    import random
    # rand1 = random.randint(0, 3)  # [0,3]
    # rand2 = random.randint(0, 3)
    # seq = sub_seq[rand1] + seq + sub_seq[rand2]
    kmer = [seq[x:x + k] for x in range(seq_length - k + 1)]
    # kmer = ['CLS'] + kmer + ['SEP']
    # print(len(kmer))
    kmers = " ".join(kmer)
    return kmers


def circRNA_Bert(dataloader, model, tokenizer, device):
    features = []
    seq = []
    # tokenizer = BertTokenizer.from_pretrained("/home/zhuhaoran/PrismNet-master/3-new-12w-0", do_lower_case=False)
    # model = BertModel.from_pretrained("/home/zhuhaoran/PrismNet-master/3-new-12w-0/")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # model = torch.nn.DataParallel(model)
    # model = model.eval()
    for sequences in dataloader:
        seq.append(sequences)

        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding='max_length')
        # print(ids)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        token_type_ids = torch.tensor(ids['token_type_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        # print(attention_mask)
        with torch.no_grad():
            # embedding = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)[0]
            embedding = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
            # print(embedding.shape)
        embedding = embedding.cpu().numpy()

        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            # print(embedding[0].shape)
            # seq_emd = embedding[seq_num][0:seq_len]
            # seq_emd = embedding[seq_num][0]  # 只取cls画图
            # seq_emd = seq_emd.mean(0)
            seq_emd = embedding[seq_num][1:seq_len - 1]
            # print(seq_emd)
            features.append(seq_emd)
    return features


def circRNABert(protein, model, tokenizer, device, k):
    """
    file_positive_path = '/home/wangyansong/Result/dataset/' + protein + '/positive'
    file_negative_path = '/home/wangyansong/Result/dataset/' + protein + '/negative'
    sequences_pos = read_fasta(file_positive_path)
    sequences_neg = read_fasta(file_negative_path)
    #sequences1 = sequences_pos
    sequences1 = sequences_pos + sequences_neg
    """
    sequences1 = protein
    sequences = []
    Bert_Feature = []
    for seq in sequences1:
        seq = seq.strip()
        ss = seq2kmer_bert(seq, k)
        # print(len(ss))
        # sequences.append(re.sub(r"[UZOB]", "X"," ".join(re.findall(".{1}",i.upper()))))
        sequences.append(ss)
    # print(sequences)
    # sequences = myDataset()
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=2048, shuffle=False)
    Features = circRNA_Bert(dataloader, model, tokenizer, device)
    # print(Features)
    # print(len(Features))
    for i in Features:
        Feature = np.array(i)
        Bert_Feature.append(Feature)
    bb = np.array(Bert_Feature)
    # data = np.pad(bb, ((0,0),(0,2),(0,0)), 'constant', constant_values=0)
    data = bb
    return data
