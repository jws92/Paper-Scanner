import cv2
import numpy as np

import json
import os
from collections import OrderedDict

def extractor():
  label_path = "./labels"
  os.makedirs(label_path, exist_ok=True)

  raw_json_file = "./coord.json"
  with open(raw_json_file, 'r', encoding='UTF8') as jf:
    json_data = json.load(jf)

  for _, label_attr in json_data.items():
    img_filename = label_attr['filename']
    labels = label_attr['regions']
    
    coord = []
    for label in labels:
      cx = label['shape_attributes']['cx']
      cy = label['shape_attributes']['cy']
      coord.append([cx, cy])

    save_dict = {
      'filename': img_filename,
      'coordinates': coord
    }
    save_label = OrderedDict()
    save_label['label_info'] = save_dict

    save_label_json = os.path.splitext(img_filename)[0] + ".json"
    with open(os.path.join(label_path, save_label_json), 'w') as save_jf:
      json.dump(save_label, save_jf)
    


if __name__ == "__main__":
  extractor()