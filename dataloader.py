import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as trasnforms

import cv2
from PIL import Image
import numpy as np

import os, json
from natsort import natsorted

tfm = trasnforms.Compose([
        trasnforms.ToTensor(),
        trasnforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                             std=[0.2023, 0.1994, 0.2010])
      ])

class ScannerDatasets(Dataset):
  def __init__(self, img_path, label_path, tfm=None):
    self.img_path = img_path
    self.label_path = label_path
    self.tfm = tfm

    self.label_list = natsorted(os.listdir(label_path))
    self.img_list = natsorted(os.listdir(img_path))

  def __len__(self):
    assert len(self.img_list) == len(self.label_list)
    return len(self.label_list)

  def __getitem__(self, idx):
    img_file = os.path.join(self.img_path, self.img_list[idx])
    label_file = os.path.join(self.label_path, self.label_list[idx])    

    img = cv2.imread(img_file)
    with open(label_file, 'r') as jf:
      label = json.load(jf)['label_info']['coordinates']

    h, w = img.shape[:2]
    img_size = np.array([h, w], dtype=np.float32)

    img_tensor = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_tensor = Image.fromarray(img_tensor)

    label_tensor = np.array(label, dtype=np.float32)
    label_tensor = np.roll(label_tensor, shift=1, axis=1) / img_size

    if self.tfm is not None:
      img_tensor = self.tfm(img_tensor)
    else:
      img_tensor = tfm(img_tensor)
      label_tensor = torch.from_numpy(label_tensor)

    return img_tensor, label_tensor


if __name__ == "__main__":
  datasets = ScannerDatasets(img_path='./data/imgs', label_path='./data/labels', tfm=None)
  train_dataloader = DataLoader(datasets, batch_size=16, shuffle=True)

  data, label = next(iter(train_dataloader))
  print(data.shape, label)