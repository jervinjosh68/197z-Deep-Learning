import utils
from dataset import DrinksData
from engine import evaluate
import torchvision
import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
from torch.utils.data import DataLoader
import numpy as np
import transforms
from torch.optim.lr_scheduler import StepLR
import pickle
import requests

def get_file(url,path,filename, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

if __name__ == '__main__':
    
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False)
    num_classes = 4
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    dataset_test = DrinksData(os.getcwd(), transform = transforms.Compose([transforms.ToTensor()]), train = False)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0,collate_fn=utils.collate_fn)
    if not os.path.exists('weights.pth'):
        print("Downloading...")
        get_file("https://github.com/jervinjosh68/197z-Object-Detection/releases/download/v1/weights.pth", 'weights.pth',"weights.pth")
        print("weights.pthh downloaded")
    else:
        print('Specified directory already downloaded. Skipping this step.')
    pretrained_file = "fasterrcnn_mobilenet_v3_large_320_fpn_finetuned_drinks.pth"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weights = torch.load("weights.pth")

    model.load_state_dict(weights['state_dict'])
    
    model.to(device)
    evaluate(model, data_loader_test, device=device)