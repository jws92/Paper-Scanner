import torch
import torch.cuda.amp as amp
import torchvision.transforms as transforms
from model import ResidualVGG

import cv2
from PIL import Image
import numpy as np

import os, time
from natsort import natsorted

@torch.no_grad()
def inference():
  tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
  ])

  res_vgg_layer = [
    [3, 64, 2],  # in_channels, out_channels, num_modules
    [64, 128, 2],  # in_channels, out_channels, num_modules
    [128, 256, 3],  # in_channels, out_channels, num_modules
    [256, 512, 3],  # in_channels, out_channels, num_modules
    [512, 512, 3],  # in_channels, out_channels, num_modules
  ]

  model = ResidualVGG(layer_properties=res_vgg_layer).cuda()
  model.load_state_dict(torch.load('./weights/best_weight.pth'), strict=False)

  model.eval()
  dummy = torch.randn(1, 3, 224, 224).cuda()
  _ = model(dummy)

  # img_root = "./test_img/"
  img_root = "./data/imgs"
  img_filelist = natsorted(os.listdir(img_root))

  for img_file in img_filelist:
    img_filepath = os.path.join(img_root, img_file)

    img = cv2.imread(img_filepath)
    img_copy = img.copy()

    h, w = img.shape[:2]
    img_size = np.array([h, w], dtype=np.float32)

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(img)
    img = tfm(img).unsqueeze(0).cuda()

    s = time.time()
    with amp.autocast():
      outs = model(img)
    e = time.time()
    print("Elapsed time: %.6f sec" % (e - s))

    outs = outs.squeeze().detach().cpu().numpy()
      
    outs *= img_size
    outs = outs.astype(np.float32)
    outs = np.roll(outs, shift=1, axis=1)

    w1 = abs(outs[0] - outs[1])
    w2 = abs(outs[2] - outs[3])
    h1 = abs(outs[0] - outs[2])
    h2 = abs(outs[1] - outs[3])

    width = max(w1[0], w2[0])
    height = max(h1[1], h2[1])

    warp_points = np.float32([
      [0, 0], [width-1, 0], [width-1, height-1], [0, height-1]
    ])

    mtx = cv2.getPerspectiveTransform(outs, warp_points)
    result = cv2.warpPerspective(img_copy, mtx, (int(width), int(height)))

    for out_coord in outs:
        cv2.circle(img_copy, (int(out_coord[0]), int(out_coord[1])), 5, (0, 255, 0), 2)

    cv2.imshow("original", img_copy)
    cv2.imshow("res", result)
    if cv2.waitKey() == 27: break


if __name__ == "__main__":
  inference()