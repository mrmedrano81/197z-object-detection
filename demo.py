import os
import numpy as np
import torch
import torch.utils.data
import cv2
from train import custom_model_function
import imutils
from datadl import download_url

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using GPU: ", torch.cuda.is_available())


#download and load pretrained_model.pth
if not os.path.exists('pretrained_model.pth'):
    download_url("https://github.com/mrmedrano81/197Z-assignment-2/releases/download/v1.0/pretrained_model.pth", "pretrained_model.pth")
else:
    print('pretrained_model.pth already exists, skipping download...')

num_classes = 4
loaded_model = custom_model_function(num_classes)
loaded_model.load_state_dict(torch.load("pretrained_model.pth"))
loaded_model.eval()

#initializing live capture
cam = cv2.VideoCapture(0)
while True:
    check, frame = cam.read()
    original = frame.copy()

    #convert frame to tensor format for input to model
    frame = imutils.resize(frame, width=640, height=480)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2,0,1))
    frame = np.expand_dims(frame, axis=0)
    frame = frame/255
    frame = torch.tensor(frame, dtype=torch.float32)

    with torch.no_grad():
      prediction = loaded_model(frame)

    #extract necessary information for bounding boxes and corresponding labels
    tensor_bboxes = prediction[0].get('boxes')
    tensor_labels = prediction[0].get('labels')

    for i in range(tensor_bboxes.size(dim=0)):
      tensor_bbox = tensor_bboxes[i]
      label_id = tensor_labels[i]
      xmin = int(torch.IntTensor.item(tensor_bbox[0]))
      ymin = int(torch.IntTensor.item(tensor_bbox[1]))
      xmax = int(torch.IntTensor.item(tensor_bbox[2]))
      ymax = int(torch.IntTensor.item(tensor_bbox[3]))

      text = ''
      color = ()
      if label_id == 1:
        text = 'Summit'
        color = (234, 40, 27)
      if label_id == 2:
        text = 'Coca Cola'
        color = (97, 37, 216)
      if label_id == 3:
        text = 'Del Monte'
        color = (94, 224, 232)

      cv2.rectangle(original, (xmin, ymax), (xmax, ymin), color, 2)
      cv2.putText(original, text, (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    cv2.imshow('demo [press "Esc" key to exit]', original)

    key = cv2.waitKey(1)

    #enter Esc key to exit
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()