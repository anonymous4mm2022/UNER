# coding=utf-8
"""
划分原始数据为train,dev,test
"""
import os
from gensim.models import Word2Vec
def load_img_id_lst(text_file_path):
    img_id_lst = []
    for line in open(text_file_path,'r',encoding='utf-8'):
        if line.startswith("IMGID:"):
            img_id_lst.append(int(line.replace("IMGID:", "").strip()))

    random.shuffle(img_id_lst)
    return  img_id_lst

# paths
data_dir = 'data/data'
text_file_path = '{}/text.txt'.format(data_dir)
    
# texts 
texts = []
img_id_lst = []
words = []
for line in open(text_file_path,'r',encoding='utf-8'):
    if line.startswith("IMGID:"):
        if len(img_id_lst)>0:
            img_id = img_id_lst[-1] # ensure img id
            texts.append(words)
        img_id_lst.append(int(line.replace("IMGID:", "").strip()))
        words = [] # clear words
    else:
        if len(line.strip())>0:
            words.append(line.strip().split(' ')[0].strip().lower())

print(len(texts))    

# domains
domain_file_path = '{}/domain.csv'.format(data_dir)
for line in open(domain_file_path,'r',encoding='utf-8'):
    if line.startswith("IMGID:"):
        imgid,domain = line.strip().split(',')
        texts.append(domain.strip().split('.'))
print(len(texts))   
# OCRs
from utils import get_ocr_texts
ocr_texts = get_ocr_texts('{}/imgs_ocr'.format(data_dir))
for key in ocr_texts.keys():
    texts.extend(ocr_texts[key])
print(len(texts)) 
# word2vec
dim = 512
model = Word2Vec(sentences=texts, vector_size=dim, window=1, min_count=1, workers=4)
model.wv.save_word2vec_format('word_vector_{}d.vec'.format(dim))
