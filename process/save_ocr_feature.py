import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import joblib
import numpy as np
import os
IMAGE_DIR = 'data/data/imgs_np'

# 读取图像路径
img_lst_ = os.listdir(IMAGE_DIR)
img_lst = ['{}/{}'.format(IMAGE_DIR,img) for img in img_lst_ if '.npy' in img]
imgid_lst = [int(img.strip().split('.')[0]) for img in img_lst_ if '.npy' in img ]

# 读取OCR结果
from utils import get_ocr_texts
texts = get_ocr_texts('data/data/imgs_ocr')

data_dir = 'data/data'

# word2vec
## load word2vec
embedding_index={}
with open('word_vector_512d.vec', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs


text_features={}
max_lens = []
cnt_no = 0
for i in tqdm(range(len(imgid_lst))):
    img_id = imgid_lst[i]
    if img_id in texts.keys():
        text = texts[img_id]
        embs_ = []
        for words in text:
            for word in words:
                if word in embedding_index.keys():
                    embs_.append(embedding_index[word])
                else:
                    embs_.append([-1]*512)
        if len(embs_)<10:
            for i in range(10-len(embs_)):
                embs_.append([-1]*512)
        embs_=embs_[:10]
    else:
        #print(img_id)
        cnt_no = cnt_no+1
        embs_=[[-1]*512]*10
    embs_ = np.asarray(embs_,dtype='float32')
    embs = torch.tensor(embs_)
    #print(embs.shape)
    text_features[img_id]=embs
print(cnt_no)
torch.save(text_features,'{}/img_ocr_features.pt'.format(data_dir))
