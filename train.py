import utils
from dataset import DrinksData
from engine import train_one_epoch, evaluate
import torchvision
import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
from torch.utils.data import DataLoader
import numpy as np
import transforms
from torch.optim.lr_scheduler import StepLR


# load a model pre-trained pre-trained on COCO

if __name__ == '__main__':
    # split the dataset in train and test set
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)


    num_classes = 4 
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    dataset = DrinksData(os.getcwd(), transform = transforms.Compose([transforms.ToTensor()]), train = True)
    
    data_loader = DataLoader( dataset, batch_size=2, shuffle=True, num_workers=0,collate_fn=utils.collate_fn)
    dataset_test = DrinksData(os.getcwd(), transform = transforms.Compose([transforms.ToTensor()]), train = False)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0,collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.05)
    lr_scheduler = StepLR(optimizer,step_size=3,gamma=0.1)
    num_epochs = 12
    for epoch in range(num_epochs):
        print("Training on: ", device)
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    evaluate(model, data_loader_test, device=device)
    state = {'state_dict' : model.state_dict()}
    torch.save(state, "weights.pth")
