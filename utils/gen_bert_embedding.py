import numpy as np
import torch
import torch.utils.data

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
    kmer = [seq[x:x + k] for x in range(seq_length - k + 1)]
    kmers = " ".join(kmer)
    return kmers


def circRNA_Bert(dataloader, model, tokenizer, device):
    features = []
    seq = []
    for sequences in dataloader:
        seq.append(sequences)

        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding='max_length')
        # print(ids)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        token_type_ids = torch.tensor(ids['token_type_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
            # print(embedding.shape)
        embedding = embedding.cpu().numpy()

        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len - 1]
            # print(seq_emd.shape)
            features.append(seq_emd)
    return features

def circRNABert(protein, model, tokenizer, device, k):
    sequences1 = protein
    sequences = []
    Bert_Feature = []
    for seq in sequences1:
        seq = seq.strip()
        ss = seq2kmer_bert(seq, k)
        sequences.append(ss)
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=2048, shuffle=False)
    Features = circRNA_Bert(dataloader, model, tokenizer, device)
    for i in Features:
        Feature = np.array(i)
        Bert_Feature.append(Feature)
    bb = np.array(Bert_Feature)
    data = bb
    return data
