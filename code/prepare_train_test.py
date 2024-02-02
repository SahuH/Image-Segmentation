import os
import math
import time
from collections import defaultdict
import re
import random

import numpy as np
import pandas as pd


paths_df = pd.read_csv('../saved_files/paths_df.csv')
image_no_lst = paths_df['image_no'].values

split = 0.2
random.Random(143).shuffle(image_no_lst)
num_training = round((1-split)*len(image_no_lst))
train_ids = image_no_lst[:num_training]
test_ids = image_no_lst[num_training:]
print(len(train_ids), len(test_ids))

pd.DataFrame(train_ids, columns=['train_ids']).to_csv('../saved_files/train_ids.csv')
pd.DataFrame(test_ids, columns=['test_ids']).to_csv('../saved_files/test_ids.csv')