from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from torch import device
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import argparse
import torchvision.transforms as T
import pickle
from imutils.video import VideoStream
from imutils.video import FPS
import requests
import os

def get_file(url,path,filename, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(path, 'wb') as downloaded:
        for chunk in r.iter_content(chunk_size=chunk_size):
            downloaded.write(chunk)
CLASSES = [0,"Summit Water","Coca Cola","Del Monte Pineapple Juice"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

if not os.path.exists('weights.pth'):
    print("weights.pth does not exist. Downloading...")
    get_file("https://github.com/jervinjosh68/197z-Object-Detection/releases/download/v1/weights.pth", 'weights.pth',"weights.pth")
    print("weights.pth downloaded")
else:
    print('Specified file (weights.pth) already downloaded. Skipping this step.')
weights = torch.load("weights.pth")
model.load_state_dict(weights["state_dict"])
model.eval()
model.to(device)

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2.0)
fps = FPS().start()
    
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("DEMO.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=640, height=480)
    orig = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.FloatTensor(frame)
    frame = frame.to(device)
    with torch.inference_mode():
        detections = model(frame)[0]
    for i in range(0, len(detections["boxes"])):
        confidence = detections["scores"][i]
        if confidence > 0.7:
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(orig, (startX, startY), (endX, endY),
				COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
    if ret == True:
        out.write(orig)
        cv2.imshow('Drinks Detector Demo', orig)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
out.release()
cv2.destroyAllWindows()
