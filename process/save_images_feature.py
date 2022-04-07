#coding=utf-8
"""
save images features
"""
import torch

data_dir = 'data/data'
ocr_feature = '{}/img_ocr_features.pt'.format(data_dir)
regions_feature = '{}/img_frcn_features.pt'.format(data_dir)
global_feature = '{}/img_resnet50_features.pt'.format(data_dir)


ocr_feature = torch.load(ocr_feature)
regions_feature = torch.load(regions_feature)
global_feature = torch.load(global_feature)

assert len(ocr_feature) == len(regions_feature)

assert len(ocr_feature) == len(global_feature)

imgids = list(ocr_feature.keys())

imgs_feature={}

for imgid in imgids:
    of = ocr_feature[imgid]
    rf = regions_feature[imgid]
    gf = global_feature[imgid]
    f = torch.cat((of,rf),0)
    f = torch.cat((f,gf),0)
    imgs_feature[imgid]=f

print(imgs_feature[imgid].shape)
torch.save(imgs_feature,'{}/img_total_features.pt'.format(data_dir))
