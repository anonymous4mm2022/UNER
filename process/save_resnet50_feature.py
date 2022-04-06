
from torchvision import models 
import argparse
import torch
import os
import time
import numpy as np
import torch.nn as nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/data", type=str, help="Path for data dir")
    parser.add_argument("--img_dir", default="imgs_np", type=str, help="Path for img dir")
    parser.add_argument("--feature_file", default="img_resnet50_features.pt", type=str, help="Filename for preprocessed image features")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet50(pretrained = True)
    model.fc = nn.Linear(2048, 49*512)
    torch.nn.init.eye(model.fc.weight)
    for param in model.parameters():
        param.requires_grad = False

    #print(model.layer4)
    model.to(device)
    model.eval()

    # Only load the images that is in train/dev/test
    img_id_lst = []
    for text_filename in ['text.txt']:
        with open(os.path.join(args.data_dir, text_filename), 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("IMGID:"):
                    img_id_lst.append(int(line.replace("IMGID:", "").strip()))

    mean_pixel = [103.939, 116.779, 123.68]  # From original code setting

    img_features = {}
    cur_time = time.time()
    for idx, img_id in enumerate(img_id_lst):
        #img_path = os.path.join(args.data_dir, args.img_dir, '{}.jpg'.format(img_id))
        img_path = os.path.join(args.data_dir, args.img_dir, '{}.npy'.format(img_id))
        try:
            """im = Image.open(img_path)
            im = im.resize((224, 224))
            im = np.array(im)

            if im.shape == (224, 224):  # Check whether the channel of image is 1
                im = np.concatenate((np.expand_dims(im, axis=-1),) * 3, axis=-1)  # Change the channel 1 to 3

            im = im[:, :, :3]  # Some images have 4th channel, which is transparency value"""
            im = np.load(img_path)


        except Exception as inst:
            print("{} error!".format(img_id))
            print(inst)
            continue

        for c in range(3):
            im[:, :, c] = im[:, :, c] - mean_pixel[c]
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)
        im = torch.Tensor(im).to(device)
        with torch.no_grad():
            img_feature = model(im)
        #print(img_feature.shape)

        img_feature = img_feature.squeeze(0).view(49,512)
        #print(img_feature)
    
        img_features[img_id] = img_feature.to("cpu")  # Save as cpu

        if (idx + 1) % 100 == 0:
            print("{} done - extracted in {:.2f} sec".format(idx + 1, time.time() - cur_time))
            cur_time = time.time()

    # Save features with torch.save
    torch.save(img_features, os.path.join(args.data_dir, args.feature_file))

