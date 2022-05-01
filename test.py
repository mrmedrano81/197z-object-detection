import torch
import transforms
import utils
import engine
import torch.utils.data
import label_utils
import train 
import os
from datadl import extract_zip_files, download_url

def main():

    #download drinks dataset
    if not os.path.exists('drinks'):
        extract_zip_files("https://github.com/mrmedrano81/197z-object-detection/releases/download/v1.0/drinks.zip", 'drinks.zip', 'drinks')
    else:
        print('drinks directory already exists, skipping download...')

    #download pretrained_model.pth
    if not os.path.exists('pretrained_model.pth'):
        download_url("https://github.com/mrmedrano81/197z-object-detection/releases/download/v1.0/pretrained_model.pth", "pretrained_model.pth")
    else:
        print('pretrained_model.pth already exists, skipping download...')

    test_dict, _ = label_utils.build_label_dictionary("labels_test.csv")
    test_dataset = train.DrinksDataset(test_dict, transform=transforms.Compose([transforms.ToTensor()]))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 4
    loaded_model = train.custom_model_function(num_classes)
    loaded_model.load_state_dict(torch.load("pretrained_model.pth"))

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    engine.evaluate(loaded_model.cuda(), data_loader_test, device=device)

if __name__ == '__main__':
    main()
