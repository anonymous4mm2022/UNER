from collections import Counter
import tldextract
from tqdm import tqdm 
import pandas as pd
# 原始数据集路径
data_dir = 'data/data'
print('分析数据集：')
print('文本数据分析...')
# text
text_file_path = '{}/text.txt'.format(data_dir)

texts = []
tags = []
img_id_lst = []
words = []
tag = []
for line in open(text_file_path,'r',encoding='utf-8'):
    if line.startswith("IMGID:"):
        if len(img_id_lst)>0:
            img_id = img_id_lst[-1] # ensure img id
            texts.append(words)
            tags.append(tag)
        img_id_lst.append(int(line.replace("IMGID:", "").strip()))
        words = [] # clear words
        tag= []
    else:
        if len(line.strip())>0:
            words.append(line.strip().split(' ')[0].strip().lower())
            tag.append(line.strip().split(' ')[1].strip().upper())
print('句子数:',len(texts))
print('词数:',sum([len(words) for words in texts]))
print('标注句子数：',len([tag for tag in tags if 'B-ORG' in tag]))
def cnt_tag(tag):
    cnt = 0
    for t in tag:
        if t in ['B-ORG']:
            cnt = cnt + 1
    return cnt

print('标注词数：',sum([cnt_tag(tag) for tag in tags if 'B-ORG' in tag ]))

print('句子长度分布:')
text_len_dict = dict(Counter([len(words) for words in texts]))
print(text_len_dict)
print('句子标注数分布')
text_tag_dict = dict(Counter([cnt_tag(tag) for tag in tags if 'B-ORG' in tag ]))
print(text_tag_dict)

# dns
print('DNS数据分析...')
dns_file_path = '{}/domain.csv'.format(data_dir)
df = pd.read_csv(dns_file_path)
domains = list(set(df['domain'].values))
print('DNS句子数：{}'.format(len(domains)))
tlds = []
for i in tqdm(range(len(domains))):
    if len(tldextract.extract(domains[i]).suffix)==0:
        print(domains[i])
    tlds.append(tldextract.extract(domains[i]).suffix)
print('DNS句子长度分布：')
dns_words = [domain.strip().split('.') for domain in domains]
dns_len_dict = dict(Counter([len(words) for words in dns_words]))
print(dns_len_dict)

print('DNS句子数后缀分布：')
tld_text_dict=dict(Counter(tlds))
print(sorted(tld_text_dict.items(), key = lambda kv:(kv[1], kv[0])))

# text
text_file_path = '{}/text.contrasts.txt'.format(data_dir)
texts_ = []
tags_ = []
img_id_lst = []
words = []
tag = []
for line in open(text_file_path,'r',encoding='utf-8'):
    if line.startswith("IMGID:"):
        if len(img_id_lst)>0:
            img_id = img_id_lst[-1] # ensure img id
            texts_.append(words)
            tags_.append(tag)
        img_id_lst.append(int(line.replace("IMGID:", "").strip()))
        words = [] # clear words
        tag= []
    else:
        if len(line.strip())>0:
            words.append(line.strip().split(' ')[0].strip().lower())
            tag.append(line.strip().split(' ')[1].strip().upper())
#print(tags_)
# 比较重合的实体标注
cnt_user = 0
cnt_org = 0
cnt_tim = 0
cnt_geo = 0
cnt_per = 0
cnt_eve = 0
cnt_gpe = 0
cnt_art = 0
cnt_nat = 0
cnt_none = 0
tags__ = []
for t in tags_:
    tags__.extend(t)
tags__=list(set(tags__))
print(tags__)

for imgid,tag,tag_ in zip(img_id_lst,tags,tags_):
    if len(tag)!=len(tag_):
        print(imgid)
    for i in range(len(tag)):
        t = tag[i]
        t_ = tag_[i]
        if t in ['B-ORG','I-ORG']:
            cnt_user = cnt_user + 1
            if t_ in ['B-ORG','I-ORG']:
                cnt_org = cnt_org + 1
            if t_ in ['B-TIM','I-TIM']:
                cnt_tim = cnt_tim + 1
            if t_ in ['B-GEO','I-GEO']:
                cnt_geo = cnt_geo + 1
            if t_ in ['B-PER','I-PER']:
                cnt_per = cnt_per + 1
            if t_ in ['B-EVE','I-EVE']:
                cnt_eve = cnt_eve + 1
            if t_ in ['B-GPE','I-GPE']:
                cnt_gpe = cnt_gpe + 1    
            if t_ in ['B-ART','I-ART']:
                cnt_art = cnt_art + 1   
            if t_ in ['B-NAT','I-NAT']:
                cnt_nat = cnt_nat + 1           
            if t_ == 'O':
                cnt_none = cnt_none + 1
print('User Entity 中常见Entity分布：')
print('USER: {}'.format(cnt_user))
print('ORG: {}  {}%'.format(cnt_org,round(cnt_org/cnt_user,4)*100))
print('TIM: {}  {}%'.format(cnt_tim,round(cnt_tim/cnt_user,4)*100))
print('GEO: {}  {}%'.format(cnt_geo,round(cnt_geo/cnt_user,4)*100))
print('PER: {}  {}%'.format(cnt_per,round(cnt_per/cnt_user,4)*100))
print('EVE: {}  {}%'.format(cnt_eve,round(cnt_eve/cnt_user,4)*100))
print('GPE: {}  {}%'.format(cnt_gpe,round(cnt_gpe/cnt_user,4)*100))
print('ART: {}  {}%'.format(cnt_art,round(cnt_art/cnt_user,4)*100))
print('NAT: {}  {}%'.format(cnt_nat,round(cnt_nat/cnt_user,4)*100))
print('O: {}  {}%'.format(cnt_none,round(cnt_none/cnt_user,4)*100))

print('常见Entity中User Entity分布：')
# ORG
cnt_user = 0
cnt_entity = 0
for tag,tag_ in zip(tags,tags_):
    for i in range(len(tag)):
        t = tag[i]
        t_ = tag_[i]
        if t_ in ['B-ORG','I-ORG']:
            cnt_entity = cnt_entity + 1
            if t in ['B-ORG','I-ORG']:
                cnt_user = cnt_user + 1
print('ORG: {}'.format(cnt_entity))     
if cnt_entity > 0:   
    print('USER: {}  {}%'.format(cnt_user,round(cnt_user/cnt_entity*100,2)))

# TIM
cnt_user = 0
cnt_entity = 0
for tag,tag_ in zip(tags,tags_):
    for i in range(len(tag)):
        t = tag[i]
        t_ = tag_[i]
        if t_ in ['B-TIM','I-TIM']:
            cnt_entity = cnt_entity + 1
            if t in ['B-ORG','I-ORG']:
                cnt_user = cnt_user + 1
print('TIM: {}'.format(cnt_entity))        
if cnt_entity > 0:   
    print('USER: {}  {}%'.format(cnt_user,round(cnt_user/cnt_entity*100,2)))
# GEO
cnt_user = 0
cnt_entity = 0
for tag,tag_ in zip(tags,tags_):
    for i in range(len(tag)):
        t = tag[i]
        t_ = tag_[i]
        if t_ in ['B-GEO','I-GEO']:
            cnt_entity = cnt_entity + 1
            if t in ['B-ORG','I-ORG']:
                cnt_user = cnt_user + 1
print('GEO: {}'.format(cnt_entity))        
if cnt_entity > 0:   
    print('USER: {}  {}%'.format(cnt_user,round(cnt_user/cnt_entity*100,2)))
# PER
cnt_user = 0
cnt_entity = 0
for tag,tag_ in zip(tags,tags_):
    for i in range(len(tag)):
        t = tag[i]
        t_ = tag_[i]
        if t_ in ['B-PER','I-PER']:
            cnt_entity = cnt_entity + 1
            if t in ['B-ORG','I-ORG']:
                cnt_user = cnt_user + 1
print('PER: {}'.format(cnt_entity))        
if cnt_entity > 0:   
    print('USER: {}  {}%'.format(cnt_user,round(cnt_user/cnt_entity*100,2)))
# EVE
cnt_user = 0
cnt_entity = 0
for tag,tag_ in zip(tags,tags_):
    for i in range(len(tag)):
        t = tag[i]
        t_ = tag_[i]
        if t_ in ['B-EVE','I-EVE']:
            cnt_entity = cnt_entity + 1
            if t in ['B-ORG','I-ORG']:
                cnt_user = cnt_user + 1
print('EVE: {}'.format(cnt_entity))        
if cnt_entity > 0:   
    print('USER: {}  {}%'.format(cnt_user,round(cnt_user/cnt_entity*100,2)))
# GPE
cnt_user = 0
cnt_entity = 0
for tag,tag_ in zip(tags,tags_):
    for i in range(len(tag)):
        t = tag[i]
        t_ = tag_[i]
        if t_ in ['B-GPE','I-GPE']:
            cnt_entity = cnt_entity + 1
            if t in ['B-ORG','I-ORG']:
                cnt_user = cnt_user + 1
print('GPE: {}'.format(cnt_entity))        
if cnt_entity > 0:   
    print('USER: {}  {}%'.format(cnt_user,round(cnt_user/cnt_entity*100,2)))
# ART
cnt_user = 0
cnt_entity = 0
for tag,tag_ in zip(tags,tags_):
    for i in range(len(tag)):
        t = tag[i]
        t_ = tag_[i]
        if t_ in ['B-ART','I-ART']:
            cnt_entity = cnt_entity + 1
            if t in ['B-ORG','I-ORG']:
                cnt_user = cnt_user + 1
print('ART: {}'.format(cnt_entity))        
if cnt_entity > 0:   
    print('USER: {}  {}%'.format(cnt_user,round(cnt_user/cnt_entity*100,2)))
# NAT
cnt_user = 0
cnt_entity = 0
for tag,tag_ in zip(tags,tags_):
    for i in range(len(tag)):
        t = tag[i]
        t_ = tag_[i]
        if t_ in ['B-NAT','I-NAT']:
            cnt_entity = cnt_entity + 1
            if t in ['B-ORG','I-ORG']:
                cnt_user = cnt_user + 1
print('NAT: {}'.format(cnt_entity))        
if cnt_entity > 0:   
    print('USER: {}  {}%'.format(cnt_user,round(cnt_user/cnt_entity*100,2)))
# O
cnt_user = 0
cnt_entity = 0
for tag,tag_ in zip(tags,tags_):
    for i in range(len(tag)):
        t = tag[i]
        t_ = tag_[i]
        if t_ in ['O']:
            cnt_entity = cnt_entity + 1
            if t in ['B-ORG','I-ORG']:
                cnt_user = cnt_user + 1
print('O: {}'.format(cnt_entity))        
if cnt_entity > 0:   
    print('USER: {}  {}%'.format(cnt_user,round(cnt_user/cnt_entity*100,2)))