import os
import numpy as np
import torch
import transforms
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_320_fpn
import torch.utils.data
import cv2
import label_utils
from dataset import DrinksData
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_mobilenet_v3_large_320_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
weights = torch.load("weights2.pth")
model.load_state_dict(weights["state_dict"])
model.eval()


# pick one image from the test set
dataset_test = DrinksData("D:/GitHub/197z-Object-Detection", transform = transforms.Compose([transforms.ToTensor()]), train = False)
img = dataset_test[43]
print("Ground truth: ", img[1].get('boxes'))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# put the model in evaluation mode
with torch.no_grad():
    prediction = model([img[0]])
    print("Prediction: ", prediction[0].get('boxes'))

# define a transform to convert a tensor to PIL image
transform = torchvision.transforms.ToPILImage()

# convert the tensor to PIL image using above transform
pil_img = transform(img[0])
cv_img = np.array(pil_img) 

# Convert RGB to BGR 
cv_img = cv_img[:, :, ::-1].copy() 


tensor_bboxes = prediction[0].get('boxes')
tensor_labels = prediction[0].get('labels')

for i in range(tensor_bboxes.size(dim=0)):
  tensor_bbox = tensor_bboxes[i]
  label_id = tensor_labels[i]
  xmin = int(torch.IntTensor.item(tensor_bbox[0]))
  ymin = int(torch.IntTensor.item(tensor_bbox[1]))
  xmax = int(torch.IntTensor.item(tensor_bbox[2]))
  ymax = int(torch.IntTensor.item(tensor_bbox[3]))

  print("bbox coords: ", xmin," ", ymin," ", xmax," ", ymax)
  original = cv_img.copy()

  text = ''
  color = ()
  if label_id == 1:
    text = 'Summit'
    color = (234, 40, 27)
  if label_id == 2:
    text = 'Coca Cola'
    color = (97, 37, 216)
  if label_id == 3:
    text = 'Pineapple Juice'
    color = (94, 224, 232)

  cv2.rectangle(cv_img, (xmin, ymax), (xmax, ymin), color, 2)
  cv2.putText(cv_img, text, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

cv2.imshow("prediction", cv_img)
cv2.waitKey()