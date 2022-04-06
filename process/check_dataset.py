#coding=utf-8
"""
根据标准的domain.csv和text.txt补充imgs
"""
import pandas as pd 
from shutil import copyfile
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import shutil
# 原始数据集路径
data_dir = 'data/data'
# text
text_file_path = '{}/text.txt'.format(data_dir)
img_id_lst = []
img_id_lst1 = []
for line in open(text_file_path,'r',encoding='utf-8'):
    if line.startswith("IMGID:"):
        img_id_lst.append(line.strip())
        img_id_lst1.append(int(line.strip().replace('IMGID:','').strip()))
# 检查格式
print('检查TEXT文件格式：')
for line in open(text_file_path,'r',encoding='utf-8'):
    if line.startswith("IMGID:"):
        img_id=int(line.strip().replace('IMGID:','').strip())
    if len(line.strip().split(' '))==1 and not line.startswith("IMGID:") and len(line.strip())>0:
        print('img_id:',img_id)
        print(line)



print('重复文本的imgid:')
imgid_text_dict={}
for imgid in img_id_lst:
    if imgid in imgid_text_dict.keys():
        print(imgid)
    else:
        imgid_text_dict[imgid]=1

# domain
text_file_path = '{}/domain.csv'.format(data_dir)
df = pd.read_csv(text_file_path)
def get_pure_domain(domain):
    return domain.replace('https://','').replace('http://','').replace('/','').replace('www.','').strip()
df['domain'] = df.apply(lambda x:get_pure_domain(x['domain']),axis=1)
#df.drop(columns=['num_id'],inplace=True)
def get_imgid(imgid):
    return int(imgid.replace('IMGID:',''))
df['intid'] = df.apply(lambda x:get_imgid(x['id']),axis=1)
df.sort_values(by=['intid'],inplace=True,ascending=False)
df.drop(columns=['intid'],inplace=True)
df.to_csv(text_file_path,index=0)
imgid_domain_dict={}
domain_imgid_dict={}
print('重复domain的imgid:')
for index,row in df.iterrows():
    imgid=row['id']
    domain=row['domain']
    pure_domain = get_pure_domain(domain)
    if imgid in imgid_domain_dict.keys():
        print(imgid)
    else:
        imgid_domain_dict[imgid]=domain
    int_imgid = int(imgid.replace('IMGID:','').strip())
    if pure_domain in  domain_imgid_dict.keys():
        domain_imgid_dict[pure_domain].append(int_imgid)
    else:
        domain_imgid_dict[pure_domain]=[int_imgid]

domains_undo=[]

# 未完成的domain
cnt = 0
for domain in domain_imgid_dict.keys():
    if len(set(domain_imgid_dict[domain]))==1:
        domains_undo.append(domain)
        cnt = cnt+1
print('还需增加数据的domains数：',cnt)

need_domains_path = 'domains.csv'

if os.path.exists(need_domains_path):
    os.remove(need_domains_path)
if cnt > 0:
    f = open(need_domains_path,'w',encoding='utf-8')
    f.write('domain\n')
    for domain in domains_undo:
        f.write('http://{}\n'.format(domain))
    f.close()

# unlabeled domain
print('未标记的domain:')
for key in imgid_domain_dict.keys():
    if key not in imgid_text_dict.keys():
        print(key)
for key in imgid_text_dict.keys():
    if key not in imgid_domain_dict.keys():
        print(key)


# images
img_dir = '{}/imgs'.format(data_dir)
for key in domain_imgid_dict.keys():
    img_path=None
    noexist_img_paths = []
    imgids = domain_imgid_dict[key]
    for imgid in imgids:
        #print(imgid)
        if os.path.exists('{}/{}.jpg'.format(img_dir,imgid)):
            img_path = '{}/{}.jpg'.format(img_dir,imgid)
        else:
            noexist_img_paths.append('{}/{}.jpg'.format(img_dir,imgid))
    if img_path is not None:
        for noexist_img_path in noexist_img_paths:
            copyfile(img_path,noexist_img_path)
    else:
        print('domain:{} imgs not exists!'.format(key))

print('domains总数：{}'.format(len(domain_imgid_dict)))
print('img总数:{}'.format(len(imgid_domain_dict)))
print('text总数:{}'.format(len(imgid_text_dict)))

img_dir = '{}/imgs'.format(data_dir)
img_files = os.listdir(img_dir)
img_files = [img_file for img_file in img_files if 'jpg' in img_file]
img_id_lst2 = [int(img_file.strip().split('.')[0]) for img_file in img_files]

print('缺少的图像ID：')
whole_img_id_lst = list(range(0,max(img_id_lst2)+1))
lack_img_id_lst = [id for id in whole_img_id_lst if id not in img_id_lst2]
f = open('lack_img_id.txt','w',encoding='utf-8')
for id in lack_img_id_lst:
    f.write('{}\n'.format(id))
f.close()

# total imgs
img_lst = os.listdir(img_dir)

total_img_id_lst = [int(img.strip().split('.')[0]) for img in img_lst]
img_id_lst_ = [id for id in total_img_id_lst if id not in img_id_lst1]

print('未标注图像个数：',len(img_id_lst_))
print('未标注图像ID：')
print(img_id_lst_)


print('转化图谱格式为Numpy格式：')

img_dir = '{}/imgs'.format(data_dir)
numpy_dir = '{}/imgs_np'.format(data_dir)
if os.path.exists(numpy_dir):
    shutil.rmtree(numpy_dir)
    os.makedirs(numpy_dir)


imgs_lst = os.listdir(img_dir)
imgs_path_lst = ['{}/{}'.format(img_dir,img) for img in img_lst if img.strip().split('.')[1]=='jpg']
imgs_np_lst = ['{}/{}.npy'.format(numpy_dir,img.strip().split('.')[0]) for img in img_lst if img.strip().split('.')[1]=='jpg']
for i in tqdm(range(len(imgs_path_lst))):
    imgs_path = imgs_path_lst[i]
    imgs_np_path = imgs_np_lst[i]
    im = Image.open(imgs_path).convert('RGB')
    #im = im.resize((224, 224))
    im = np.array(im)
    im = im[:, :, :3]  # Some images have 4th channel, which is transparency value
    # numpy 保存
    #print(im.shape)
    #print(im)
    np.save(imgs_np_path,im)



