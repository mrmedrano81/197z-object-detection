import os
import numpy as np
import torch
import transforms
from torchvision.models.detection.faster_rcnn import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
import engine
import torch.utils.data
from PIL import Image
import label_utils
from datadl import extract_zip_files


#custom model function for the fasterrcnn_mobilenet_v3_large_320_fpn
def custom_model_function(num_classes):

    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


#custom function for integrating label_utils.build_label_dictionary into dataset
def adapt_label_dict(boxes):

    labels = []
    bxs = []
    areas = []

    #list for reordering coordinates to fit model input
    order = [0, 2, 1, 3]

    for box in boxes:
        labels.append(box[-1].astype(np.int64))
        box = [box[i] for i in order]
        areas.append((box[2]-box[0])*(box[3]-box[1]))
        bxs.append(box)
    
    return bxs, labels, areas

class DrinksDataset(torch.utils.data.Dataset):
    def __init__(self, dictionary, transform=None):
        self.dictionary = dictionary
        self.transform = transform
    def __getitem__(self, idx):
        key = list(self.dictionary.keys())[idx]
        boxes = self.dictionary[key]

        img_path = os.path.join("drinks", key)
        img = Image.open(img_path)

        bxs, labels, areas = adapt_label_dict(boxes)

        idx_lst = []
        idx_lst.append(idx)

        #convert target parameters to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(bxs, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        image_id = torch.tensor(idx_lst)

        #assuming iscrowd is 0
        iscrowd = torch.zeros(len(bxs), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] =iscrowd

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.dictionary)


if __name__ == "__main__":

    #download drinks dataset
    if not os.path.exists('drinks'):
        extract_zip_files("https://github.com/mrmedrano81/197Z-assignment-2/releases/download/v1.0/drinks.zip", 'drinks.zip', 'drinks')
    else:
        print('drinks directory already exists, skipping download...')

    train_dict, _ = label_utils.build_label_dictionary("labels_train.csv")
    train_dataset = DrinksDataset(train_dict, transform=transforms.Compose([transforms.ToTensor()]))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 4
    model = custom_model_function(num_classes)
    model.to(device)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):

        engine.train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)

        lr_scheduler.step()
    
    FILE = "model.pth"
    torch.save(model.state_dict(), FILE)