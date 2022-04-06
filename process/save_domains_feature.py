import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import joblib
import numpy as np
# 原始数据集路径
data_dir = 'data/data'
# word2vec
## load word2vec
embedding_index={}
with open('bert_vector_768d.vec', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        #print(coefs.shape)
        embedding_index[word] = coefs

dns_features={}
max_lens = []

f = open('{}/domain.csv'.format(data_dir),'r',encoding='utf-8')
lines = f.readlines()
f.close()
for i in tqdm(range(1,len(lines))):
    line = lines[i]
    img_id = int(line.strip().split(',')[0].replace("IMGID:", "").strip())
    domain = line.strip().split(',')[1].strip()
    words = domain.strip().split('.')
    embs_ = []
    for word in words:
        if word in embedding_index.keys():
            embs_.append(list(embedding_index[word]))
    max_lens.append(len(embs_))
    #print(embs_)
    if len(embs_)<5:
        for i in range(5-len(embs_)):
            embs_.append([-1]*768)
    #print(embs_)
    embs_ = np.asarray(embs_,dtype='float32')
    embs = torch.tensor(embs_)
    #print(embs.shape)
    dns_features[img_id]=embs
    

print(max(max_lens))
#joblib.dump(dns_features,'{}/dns_bert_features.pkl'.format(data_dir))
torch.save(dns_features,'{}/dns_bert_features.pt'.format(data_dir))

# word2vec
## load word2vec
embedding_index={}
with open('word_vector_200d.vec', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs


dns_features={}
max_lens = []

f = open('{}/domain.csv'.format(data_dir),'r',encoding='utf-8')
lines = f.readlines()
f.close()
for i in tqdm(range(1,len(lines))):
    line = lines[i]
    img_id = int(line.strip().split(',')[0].replace("IMGID:", "").strip())
    domain = line.strip().split(',')[1].strip()
    words = domain.strip().split('.')
    embs_ = []
    for word in words:
        if word in embedding_index.keys():
            embs_.append(embedding_index[word])
    max_lens.append(len(embs_))
    if len(embs_)<5:
        for i in range(5-len(embs_)):
            embs_.append([-1]*200)

    embs_ = np.asarray(embs_,dtype='float32')
    embs = torch.tensor(embs_)
    #print(embs.shape)
    dns_features[img_id]=embs
    

print(max(max_lens))
#joblib.dump(dns_features,'{}/dns_bert_features.pkl'.format(data_dir))
torch.save(dns_features,'{}/dns_w2v_features.pt'.format(data_dir))
