import os
import math
import time
from collections import defaultdict
import re

import numpy as np
import pandas as pd
from glob import glob


paths = defaultdict(list)

names_lst = []

# Load Covid Images' paths

DIR = "../COVID-19_Radiography_Dataset"
covid_image_paths = glob(
    os.path.join(DIR,"COVID/images/*.png"))
covid_mask_paths = glob(
    os.path.join(DIR,"COVID/masks/*.png"))

for img_path in covid_image_paths:
    img_name = img_path.split('/')[-1]
    img_no = img_name.split('.')[0]
    names_lst.append(img_no)
    for mask_path in covid_mask_paths:
        mask_match = re.search(img_name, mask_path)
        if mask_match:
            paths["image_no"].append(img_no)
            paths["image_path"].append(img_path)
            paths["mask_path"].append(mask_path)

            
# Load Normal Images' paths

normal_image_paths = glob(
    os.path.join(DIR,"Normal/images/*.png"))
normal_mask_paths = glob(
    os.path.join(DIR,"Normal/masks/*.png"))

for img_path in normal_image_paths:
    img_name = img_path.split('/')[-1]
    img_no = img_name.split('.')[0]
    names_lst.append(img_no)
    for mask_path in normal_mask_paths:
        mask_match = re.search(img_name, mask_path)
        if mask_match:
            paths["image_no"].append(img_no)
            paths["image_path"].append(img_path)
            paths["mask_path"].append(mask_path)            
            
paths_df = pd.DataFrame.from_dict(paths)
paths_df['image_no'] = paths_df['image_no'].astype(str)
paths_df.to_csv('../saved_files/paths_df.csv', index=False)