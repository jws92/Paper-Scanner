import cv2
import json
import os
from natsort import natsorted

def checker():
  img_root = './imgs'
  label_root = './labels'

  label_filelist = natsorted(os.listdir(label_root))

  for label_file in label_filelist:
    with open(os.path.join(label_root, label_file), 'r') as jf:
      json_data = json.load(jf)['label_info']
    
    img_filename = json_data['filename']
    coordinates = json_data['coordinates']

    img = cv2.imread(os.path.join(img_root, img_filename))
    for coord in coordinates:
      cv2.circle(img, (coord[0], coord[1]), 5, (0, 255, 0), 2)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('img', 960, 540)
    cv2.imshow('img', img)
    if cv2.waitKey() == 27: break



if __name__ == "__main__":
  checker()