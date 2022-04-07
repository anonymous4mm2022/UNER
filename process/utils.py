#coding=utf-8
"""
some useful functions
"""
import os
import json

def get_ocr_files(ocr_dir):
    ocr_files_lst = os.listdir(ocr_dir)
    ocr_files_lst = [ocr_file for ocr_file in ocr_files_lst if '.json' in ocr_file]
    return ocr_files_lst

def get_ocr_texts(ocr_dir):
    ocr_files_lst = get_ocr_files(ocr_dir)
    texts = {}
    for ocr_file in ocr_files_lst:  
        ocr_file_path = '{}/{}'.format(ocr_dir,ocr_file)
        IMGID=int(ocr_file.strip().split('.')[0])
        f = open(ocr_file_path,'r',encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            try:
                line = json.loads(line)
                for word_result in line['words_result']:
                    if IMGID in texts.keys():
                        texts[IMGID].append(word_result['words'])
                    else:
                        texts[IMGID] = [word_result['words']]
            except Exception as e:
                print(str(e))
    print('OCR texts results:',len(texts))
    return texts
def get_ocr_boxes(ocr_dir):
    ocr_files_lst = get_ocr_files(ocr_dir)
    boxes = {}
    for ocr_file in ocr_files_lst:
        ocr_file_path = '{}/{}'.format(ocr_dir,ocr_file)
        IMGID=int(ocr_file.strip().split('.')[0])
        f = open(ocr_file_path,'r',encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            try:
                line = json.loads(line)
                #IMGID = line['IMGID']
                for word_result in line['words_result']:
                    box = [word_result['location']['left'],word_result['location']['top'],word_result['location']['width'],word_result['location']['height']]
                    if IMGID in boxes.keys():
                        boxes[IMGID].append(box)
                    else:
                        boxes[IMGID] = [box]
            except Exception as e:
                print(str(e))
    print('OCR boxes results:',len(boxes))
    return boxes


if __name__=='__main__':
    texts = get_ocr_texts('data/data/imgs_ocr')
    print(len(texts))


