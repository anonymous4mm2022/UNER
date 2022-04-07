# Copyright (c) Facebook, Inc. and its affiliates.

# install `vqa-maskrcnn-benchmark` from
# https://github.com/ronghanghu/vqa-maskrcnn-benchmark-m4c
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from PIL import Image


def load_detection_model(yaml_file, yaml_ckpt):
    cfg.merge_from_file(yaml_file)
    cfg.freeze()

    model = build_detection_model(cfg)
    checkpoint = torch.load(yaml_ckpt, map_location=torch.device("cpu"))

    load_state_dict(model, checkpoint.pop("model"))

    model.to("cuda")
    model.eval()
    return model


def _image_transform(image_path):
    #img = Image.open(image_path)
    img = np.load(image_path)
    #print(img)
    im = np.array(img).astype(np.float32)
    # handle a few corner cases
    if im.ndim == 2:  # gray => RGB
        im = np.tile(im[:, :, None], (1, 1, 3))
    if im.shape[2] > 3:  # RGBA => RGB
        im = im[:, :, :3]

    im = im[:, :, ::-1]  # RGB => BGR
    im -= np.array([102.9801, 115.9465, 122.7717])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(800) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > 1333:
        im_scale = float(1333) / float(im_size_max)
    im = cv2.resize(
        im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
    )
    img = torch.from_numpy(im).permute(2, 0, 1)
    return img, im_scale


def _process_feature_extraction(output, im_scales, feat_name="fc6"):
    batch_size = len(output[0]["proposals"])
    n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
    score_list = output[0]["scores"].split(n_boxes_per_image)
    score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
    feats = output[0][feat_name].split(n_boxes_per_image)
    cur_device = score_list[0].device

    feat_list = []
    bbox_list = []

    for i in range(batch_size):
        dets = output[0]["proposals"][i].bbox / im_scales[i]
        scores = score_list[i]

        max_conf = torch.zeros(scores.shape[0]).to(cur_device)

        for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.5)
            max_conf[keep] = torch.where(
                cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep]
            )

        keep_boxes = torch.argsort(max_conf, descending=True)[:100]
        feat_list.append(feats[i][keep_boxes])
        bbox_list.append(output[0]["proposals"][i].bbox[keep_boxes])
    return feat_list, bbox_list


def extract_features(detection_model, image_path, input_boxes=None, feat_name="fc6"):
    im, im_scale = _image_transform(image_path)
    #print(im)
    
    if input_boxes is not None:
        if isinstance(input_boxes, np.ndarray):
            input_boxes = torch.from_numpy(input_boxes.copy())
            input_boxes = input_boxes.float()
        input_boxes *= im_scale
    img_tensor, im_scales = [im], [im_scale]
    current_img_list = to_image_list(img_tensor, size_divisible=32)
    current_img_list = current_img_list.to("cuda")
    with torch.no_grad():
        #output = detection_model(current_img_list, input_boxes=input_boxes)
        output = detection_model(current_img_list, targets=input_boxes)
        #print(len(output))
    if input_boxes is None:
        feat_list, bbox_list = _process_feature_extraction(output, im_scales, feat_name)
        #print(len(feat_list))
        feat = feat_list[0].cpu().numpy()
        bbox = bbox_list[0].cpu().numpy() / im_scale
    else:
        
        feat = output[0][feat_name].cpu().numpy()
        bbox = output[0]["proposals"][0].bbox.cpu().numpy() / im_scale
        #print(len(output[0]["proposals"]))
    return feat, bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detection_cfg",
        type=str,
        default = 'detectron_model.yaml',
        help="Detectron config file; download it from "
        + "https://dl.fbaipublicfiles.com/pythia/detectron_model/"
        + "detectron_model.yaml",
    )
    parser.add_argument(
        "--detection_model",
        type=str,
        default='detectron_model.pth',
        help="Detectron model file; download it"
        + " from https://dl.fbaipublicfiles.com/pythia/detectron_model/"
        + "detectron_model.pth",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default='data/data/imgs_np1',
        help="The directory containing images(npy)",
    )
    
    args = parser.parse_args()

    DETECTION_YAML = args.detection_cfg
    DETECTION_CKPT = args.detection_model
    IMAGE_DIR = args.image_dir

    print('DETECTION_YAML:',DETECTION_YAML)
    print('DETECTION_CKPT:',DETECTION_CKPT)
    print('IMAGE_DIR:',IMAGE_DIR)
    # 读取图像路径
    img_lst_ = os.listdir(IMAGE_DIR)
    img_lst = ['{}/{}'.format(IMAGE_DIR,img) for img in img_lst_ if '.npy' in img]
    imgid_lst = [int(img.strip().split('.')[0]) for img in img_lst_ if '.npy' in img ]

    # 读取OCR结果
    from utils import get_ocr_boxes
    boxes = get_ocr_boxes('data/data/imgs_ocr')
    # 加载模型
    detection_model = load_detection_model(DETECTION_YAML, DETECTION_CKPT)

    #print(help(detection_model))
    print("Faster R-CNN OCR features")
    img_features = {} 
    for i in tqdm(range(len(img_lst))):
        image_path = img_lst[i]
        img_id = imgid_lst[i]

        if img_id in boxes.keys():
            #img = Image.open(image_path)
            
            #print(image_path)
            img = np.load(image_path)
            im = np.array(img).astype(np.float32)
            
            w = im.shape[0]
            h = im.shape[1]
            #print(w,h)
            ocr_normalized_boxes = np.array(boxes[img_id])
            #print(ocr_normalized_boxes)
            ocr_boxes = ocr_normalized_boxes.reshape(-1, 4) * [w, h, w, h]

            #for ocr_box in list(ocr_boxes):
                #ocr_box=np.array([ocr_box]).reshape(-1, 4)
            
            extracted_feat, _ = extract_features(
                        detection_model, image_path, input_boxes=ocr_boxes
            )
            extracted_feats=extracted_feat.mean(axis=0).flatten()
            #regions = 3
            #extracted_feats=[]
            #for ocr_box in list(ocr_boxes):
            #    ocr_box=np.array([ocr_box]).reshape(-1, 4)
            #    extracted_feat, _ = extract_features(
            #            detection_model, image_path, input_boxes=ocr_box
            #            )
            #    #print(extracted_feat.mean(axis=0).shape)
            #    extracted_feat=extracted_feat.mean(axis=0).flatten()
            #    extracted_feats.append(extracted_feat)
            #if len(list(ocr_boxes))<regions:
            #    for i in range(regions-len(list(ocr_boxes))):
            #        extracted_feats.append(np.zeros(2048, np.float32))
            #extracted_feats=np.array(extracted_feats[:regions],np.float32)
            #print(extracted_feats.shape)
        else:    
            extracted_feats = np.zeros(2048, np.float32).flatten()
            #print(extracted_feats.shape)
        img_feature = torch.Tensor(extracted_feats).view(4,512)
        #print(img_feature)
        img_features[img_id] = img_feature
        #print(extracted_feats)
    torch.save(img_features,'data/data/img_frcn_features.pt')


if __name__ == "__main__":
    main()
