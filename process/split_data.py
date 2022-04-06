# coding=utf-8
"""
划分原始数据为train,dev,test
"""
import random
import os
import shutil
def load_img_id_lst(text_file_path):
    img_id_lst = []
    for line in open(text_file_path,'r',encoding='utf-8'):
        if line.startswith("IMGID:"):
            img_id_lst.append(int(line.replace("IMGID:", "").strip()))

    random.shuffle(img_id_lst)
    return  img_id_lst

def split_one():
    # paths
    data_dir = 'data/data'
    dataset_dir = 'data/sample'
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)
    text_file_path = '{}/text.txt'.format(data_dir)
    
    # img_id_lst
    img_id_lst = load_img_id_lst(text_file_path)
    #print(img_id_lst[:10])
    #random.shuffle(img_id_lst)
    #print(img_id_lst[:10])
    #random.shuffle(img_id_lst)
    #print(img_id_lst[:10])
    #random.shuffle(img_id_lst)
    #random.shuffle(img_id_lst)
    #random.shuffle(img_id_lst)
    #random.shuffle(img_id_lst)
    # split
    test_size = 0.35
    dev_size = 0.3
    train_size = 1 - test_size - dev_size

    test_num = int(len(img_id_lst)*test_size)
    dev_num = int(len(img_id_lst)*dev_size)
    train_num = len(img_id_lst) - test_num - dev_num
    test_img_id_lst = img_id_lst[:test_num]
    dev_img_id_lst = img_id_lst[test_num:test_num+dev_num]
    train_img_id_lst = img_id_lst[test_num+dev_num:]

    print('train number:',len(train_img_id_lst))
    print('dev number:',len(dev_img_id_lst))
    print('test number:',len(test_img_id_lst))


    train_file_path='{}/train'.format(dataset_dir)
    dev_file_path='{}/dev'.format(dataset_dir)
    test_file_path='{}/test'.format(dataset_dir)
    save_dataset(text_file_path=text_file_path,
                 train_img_id_lst=train_img_id_lst,
                 dev_img_id_lst=dev_img_id_lst,
                 test_img_id_lst=test_img_id_lst,
                 train_file_path=train_file_path,
                 dev_file_path=dev_file_path,
                 test_file_path=test_file_path)

def split_kfold():
    k = 5
    # paths
    data_dir = 'data/data'
    dataset_dir = 'data/{}_fold'.format(k)
    if os.path.exists(dataset_dir):
    	shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)
    text_file_path = '{}/text.txt'.format(data_dir)
    # img_id_lst
    img_id_lst = load_img_id_lst(text_file_path)
    # split
    for i in range(k):
    	print('{}-fold:{}'.format(k,i))
    	train_file_path='{}/train{}'.format(dataset_dir,i)
    	dev_file_path='{}/dev{}'.format(dataset_dir,i)
    	test_file_path='{}/test{}'.format(dataset_dir,i)
    	
    	test_num = int(len(img_id_lst)/k)+1
    	dev_num = 1
    	train_num = len(img_id_lst) - test_num - dev_num
    	
    	# test_img_id_lst
    	test_img_id_lst = img_id_lst[test_num*i:test_num*(i+1)].copy()
    	img_id_lst_ = [img_id for img_id in img_id_lst if img_id not in test_img_id_lst]

    	# dev_img_id_lst
    	dev_img_id_lst = img_id_lst_[:dev_num].copy()

    	# train_img_id_lst
    	train_img_id_lst = img_id_lst_[dev_num:].copy()
    	
    	print('train number:',len(train_img_id_lst))
    	print('dev number:',len(dev_img_id_lst))
    	print('test number:',len(test_img_id_lst))

    	save_dataset(text_file_path=text_file_path,
                     train_img_id_lst=train_img_id_lst,
                     dev_img_id_lst=dev_img_id_lst,
                     test_img_id_lst=test_img_id_lst,
                     train_file_path=train_file_path,
                     dev_file_path=dev_file_path,
                     test_file_path=test_file_path)


def save_dataset(text_file_path:str,
                 train_img_id_lst:list,
                 dev_img_id_lst:list,
                 test_img_id_lst,
                 train_file_path:str,
                 dev_file_path:str,
                 test_file_path:str):
    # save
    f_tr = open(train_file_path,'w',encoding='utf-8')
    f_te = open(test_file_path,'w',encoding='utf-8')
    f_de = open(dev_file_path,'w',encoding='utf-8')
    is_train = False
    is_test = False
    is_dev = False
    img_enties_dict={}
    for line in open(text_file_path,'r',encoding='utf-8'):
        if line.startswith("IMGID:"):
            img_id = int(line.replace("IMGID:", "").strip())
            if img_id in test_img_id_lst:
                is_train = False
                is_test = True
                is_dev = False
            if img_id in dev_img_id_lst:
                is_train = False
                is_test = False
                is_dev = True
            if img_id in train_img_id_lst:
                is_train = True
                is_test = False
                is_dev = False
        else:
            entity = line.strip().split(' ')[-1]
            if len(entity)>0:
                if img_id in img_enties_dict.keys():
                    img_enties_dict[img_id].append(entity)
                else:
                    img_enties_dict[img_id]=[entity]
        # check
        if len(line.strip().split(' '))==1 and not line.startswith("IMGID:") and len(line.strip())>0:
            print(img_id)
            print(line)
        if is_train:
            f_tr.write(line)
        if is_test:
            f_te.write(line)
        if is_dev:
            f_de.write(line)
    #print(img_enties_dict)
    for key in img_enties_dict.keys():
        for i in img_enties_dict[key]:
            if i=='':
                print(key)

    f_tr.close()
    f_te.close()
    f_de.close()

if __name__ == '__main__':
    split_one()
    #split_kfold()
