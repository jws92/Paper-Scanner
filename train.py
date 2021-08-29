import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn

from model import ResidualVGG

from torch.utils.data import DataLoader
from dataloader import ScannerDatasets

from torchinfo import summary

import numpy as np
import cv2
import os


if __name__ == "__main__":
  tfms = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                              std=[0.2023, 0.1994, 0.2010])

  res_vgg_layer = [
    [3, 64, 2],  # in_channels, out_channels, num_modules
    [64, 128, 2],  # in_channels, out_channels, num_modules
    [128, 256, 3],  # in_channels, out_channels, num_modules
    [256, 512, 3],  # in_channels, out_channels, num_modules
    [512, 512, 3],  # in_channels, out_channels, num_modules
  ]

  model = ResidualVGG(layer_properties=res_vgg_layer)
  model = model.cuda()
  summary(model, input_size=(1, 3, 224, 224), device='cuda')

  cudnn.benchmark = True

  datasets = ScannerDatasets(img_path='./data/imgs', label_path='./data/labels', tfm=None)
  train_dataloader = DataLoader(datasets, batch_size=8, shuffle=True)
    
  criterion = nn.SmoothL1Loss()
  optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 470], gamma=0.1)
  # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
  # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

  scaler = amp.GradScaler(enabled=True)

  model.train()
  best_loss = 100
  num_epochs = 500

  weight_path = "./weights"
  os.makedirs(weight_path, exist_ok=True)

  for epoch in range(num_epochs):

    for i, (img_tensor, label_tensor) in enumerate(train_dataloader):
      img_tensor = img_tensor.cuda()
      label_tensor = label_tensor.cuda()

      optimizer.zero_grad()

      with amp.autocast():
        out = model(img_tensor)
        loss = criterion(out, label_tensor)
      
      # loss.backward()
      # optimizer.step()
      scaler.scale(loss).backward()
      scaler.step(optimizer=optimizer)
      scaler.update()

      if i % 1 == 0:
        print(f"[{epoch+1}/{num_epochs}][{i+1}/{len(train_dataloader)}] loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}")
        if best_loss > loss.item():
          best_loss = loss.item()
          torch.save(model.state_dict(), os.path.join(weight_path, 'best_weight.pth'))

    scheduler.step()

    torch.save(model.state_dict(), os.path.join(weight_path, 'last_weight.pth'))
    print()