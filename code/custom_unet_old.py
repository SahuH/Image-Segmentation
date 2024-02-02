

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import cv2

import os
import PIL
## checking for xrays and their respective masks
from glob import glob
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from skimage import measure

import cv2
from tqdm import tqdm

import re

paths_df = pd.read_csv('paths_df.csv')

train_ids = pd.read_csv('train_ids.csv')['train_ids']
test_ids = pd.read_csv('test_ids.csv')['test_ids']
print(len(train_ids), len(test_ids))

paths_train = paths_df[paths_df['image_no'].isin(train_ids)]
paths_test = paths_df[paths_df['image_no'].isin(test_ids)]
print(paths_train.shape, paths_test.shape)

paths_df_samp = paths_df.iloc[:3000]
paths_train = paths_df_samp[paths_df_samp['image_no'].isin(train_ids)]
paths_test = paths_df_samp[paths_df_samp['image_no'].isin(test_ids)]
print(paths_train.shape, paths_test.shape)

def prepare_train_test(df = pd.DataFrame(), resize_shape = tuple(), color_mode = "rgb"):
    img_array = list()
    mask_array = list()

    for image_path in tqdm(df.image_path):
        resized_image = cv2.resize(cv2.imread(image_path),resize_shape)
        resized_image = resized_image/255.
        if color_mode == "gray":
            img_array.append(resized_image[:,:,0])
        elif color_mode == "rgb":
            img_array.append(resized_image[:,:,:])
  
    for mask_path in tqdm(df.mask_path):
        resized_mask = cv2.resize(cv2.imread(mask_path), resize_shape)
        resized_mask = resized_mask/255.
        mask_array.append(resized_mask[:,:,0])

    return img_array, mask_array

img_train, mask_train = prepare_train_test(df = paths_train, resize_shape = (256,256), color_mode = "gray")
img_test, mask_test = prepare_train_test(df = paths_test, resize_shape = (256,256), color_mode = "gray")


# from sklearn.model_selection import train_test_split
# img_train, img_test, mask_train, mask_test = train_test_split(img_array, mask_array, test_size = 0.2, random_state= 42)

img_side_size = 256
img_train = np.array(img_train).reshape(len(img_train), img_side_size, img_side_size, 1)
img_test = np.array(img_test).reshape(len(img_test), img_side_size, img_side_size, 1)
mask_train = np.array(mask_train).reshape(len(mask_train), img_side_size, img_side_size, 1)
mask_test = np.array(mask_test).reshape(len(mask_test), img_side_size, img_side_size, 1)
print(img_train.shape, mask_train.shape)
print(img_test.shape, mask_test.shape)


dim = 256

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

def bce_loss(y_true, y_pred):
    bce = K.max(y_pred,0)-y_pred * y_true + K.log(1+K.exp((-1)*K.abs(y_pred)))
    return bce

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)    
    return dice

def dc_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def inv_dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = 1 - y_true_f
    y_pred_f = 1 - y_pred_f
    inv_intersection = K.sum(y_true_f * y_pred_f)
    inv_dice = (2. * inv_intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)    
    return inv_dice

def idc_loss(y_true, y_pred):
    return 1 - inv_dice_coef(y_true, y_pred)
    
def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jaccard_coef_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred) 

def bn_act(x, act=True):
    x = tensorflow.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tensorflow.keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = tensorflow.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = tensorflow.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = tensorflow.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tensorflow.keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = tensorflow.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = tensorflow.keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = tensorflow.keras.layers.UpSampling2D((2, 2))(x)
    c = tensorflow.keras.layers.Concatenate()([u, xskip])
    return c

def ResUNet():
    f = [16, 32, 64, 128, 256]
    inputs = tensorflow.keras.layers.Input((dim, dim, 1))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = tensorflow.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    model = tensorflow.keras.models.Model(inputs, outputs)
    return model

from tensorflow.keras.losses import BinaryCrossentropy

def custom_loss(y_true, y_pred):
    cl =  bce_loss(y_true, y_pred) + dc_loss(y_true, y_pred) + idc_loss(y_true, y_pred)
    return cl

metrics = [dice_coef, jaccard_coef, 'binary_accuracy']

from tensorflow.keras.optimizers import Adam
EPOCHS = 2
model = ResUNet()
# model.compile(optimizer=Adam(lr=5*1e-4), loss='binary_crossentropy', metrics=metrics)
model.compile(optimizer=Adam(lr=5*1e-4), loss=custom_loss, metrics=metrics)
model.summary()

model_name = 'custom_unet'

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="./logs/{}_weights.best.hdf5".format(model_name)
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', #verbose=1, 
                             save_best_only=True, #mode='min', 
                             save_weights_only = True)
early = EarlyStopping(monitor="val_loss", patience=5) 

callbacks_list = [checkpoint, early]


import time
from datetime import timedelta
start = time.time()
history = model.fit(x = img_train, 
                    y = mask_train, 
                    validation_data = (img_test, mask_test), 
                    epochs = 30, 
                    batch_size = 16,
                   callbacks = callbacks_list)
end = time.time()
delta = end - start
print(str(timedelta(seconds=delta)))


import json
json_obj = json.dumps(str(history.history))
f = open("./logs/"+model_name+"_history.json","w")
f.write(json_obj)
f.close()

# import json
# f = open("./logs/"+model_name+"_history.json","w")
# tmp = json.load(f)
# f.close()
# import ast
# tmp_1 = ast.literal_eval(tmp)

model.save("./logs/"+model_name+".h5")
