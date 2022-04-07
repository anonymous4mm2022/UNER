# coding=utf-8
"""
split train,dev,test
"""
import os
import torch
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

words = []
for text in texts:
    for word in text:
        words.append(word)

words = list(set(words))
# bert vec 
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
max_seq_len=15

model = BertModel.from_pretrained(model_name)
def bert_embeddings(word):
    
    word2token={}

    input_ids = tokenizer.encode(word, add_special_tokens=True)
    input_ids = torch.tensor([input_ids])
    
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        last_hidden_states = torch.squeeze(last_hidden_states,dim=0)
        last_hidden_states = torch.mean(last_hidden_states[1:-1,:],dim=0)
        return last_hidden_states
f = open('bert_vector_768d.vec','w',encoding='utf-8')
for i in tqdm(range(len(words))):
    word = words[i]
    emb = bert_embeddings(word).numpy()
    emb = ' '.join([str(e) for e in list(emb)])
    f.write('{} {}\n'.format(word,emb))

f.close()
