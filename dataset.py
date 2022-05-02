import os
import numpy as np
import torch
from PIL import Image
import transforms
import wandb
import label_utils
from torch.utils.data import DataLoader
from PIL import Image
import requests
from zipfile import ZipFile



class DrinksData(torch.utils.data.Dataset):
    def __init__(self, root, transform, train = True):
        self.root = root
        self.transform = transform
        self.labels = {}
        if train:
            file = open(root+"/labels_train.csv",mode = "r")
        else:
            file = open(root+"/labels_test.csv",mode = "r")
        temp_list = []
        temp_key = ""
        for line in file:
            temp_list = line.split(",")
            if temp_key == temp_list[0]:
                self.labels[temp_list[0]].append(temp_list[1:])
            else:
                self.labels[temp_list[0]] = [temp_list[1:]]
                temp_key = temp_list[0]
        del self.labels["frame"]
        self.imgs = sorted(self.labels.keys())
        if not os.path.exists('drinks'):
            print("Drinks directory does not exist. Downloading...")
            get_file("https://github.com/jervinjosh68/197z-Object-Detection/releases/download/v1/drinks.zip", 'drinks.zip', 'drinks.zip', 'drinks')
            print("drinks.zip downloaded")
        else:
            print('Specified directory (drinks.zip) already downloaded. Skipping this step.')
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "drinks/drinks", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img_name = self.imgs[idx]
        

        num_objs = len(self.labels[img_name])
        boxes = []
        labels = []
        for i in range(num_objs):
            temp_list = self.labels[img_name][i]
            xmin = int(temp_list[0])
            xmax = int(temp_list[1])
            ymin = int(temp_list[2])
            ymax = int(temp_list[3])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(temp_list[4].strip()))


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = torch.as_tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), dtype = torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)    



def get_file(url,path, filename, target_dir, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    with ZipFile(filename, 'r') as zipObj:
        zipObj.extractall(target_dir)

