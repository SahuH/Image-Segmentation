import os
import math
import time
from datetime import timedelta
from collections import defaultdict
import re
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import measure
import cv2
from tqdm import tqdm
import PIL
from glob import glob

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
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

def main(args):
    data_size = args.data_size
    img_size = args.img_size
    epochs = args.epochs

    paths_df = pd.read_csv('../saved_files/paths_df.csv')

    train_ids = pd.read_csv('../saved_files/train_ids.csv')['train_ids']
    test_ids = pd.read_csv('../saved_files/test_ids.csv')['test_ids']
    print("Total Train samples: %d"%len(train_ids))
    print("Total Test samples: %d"%len(test_ids))

    if data_size == -1:
        paths_train = paths_df[paths_df['image_no'].isin(train_ids)]
        paths_test = paths_df[paths_df['image_no'].isin(test_ids)]

    else:
        paths_df_samp = paths_df.iloc[:data_size]
        paths_train = paths_df_samp[paths_df_samp['image_no'].isin(train_ids)]
        paths_test = paths_df_samp[paths_df_samp['image_no'].isin(test_ids)]

    print("Train Samples being used: %d"%paths_train.shape[0])
    print("Test Samples being used: %d"%paths_test.shape[0])

    img_train, mask_train = prepare_train_test(df = paths_train, resize_shape = (256,256), color_mode = "gray")
    img_test, mask_test = prepare_train_test(df = paths_test, resize_shape = (256,256), color_mode = "gray")

    img_train = np.array(img_train).reshape(len(img_train), img_size, img_size, 1)
    img_test = np.array(img_test).reshape(len(img_test), img_size, img_size, 1)
    mask_train = np.array(mask_train).reshape(len(mask_train), img_size, img_size, 1)
    mask_test = np.array(mask_test).reshape(len(mask_test), img_size, img_size, 1)
    print(img_train.shape, mask_train.shape)
    print(img_test.shape, mask_test.shape)

    metrics = [dice_coef, jaccard_coef, 'binary_accuracy']
    model = CustomUNet(input_size=(img_size, img_size, 1))
    model.compile(optimizer=Adam(lr=5*1e-4), loss=custom_loss, metrics=metrics)
    model.summary()

    model_name = 'custom_unet'
    weight_path="../logs/{}_weights.best.hdf5".format(model_name)
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, save_weights_only = True)
    early = EarlyStopping(monitor="val_loss", patience=5)
    callbacks_list = [checkpoint, early]

    start = time.time()
    history = model.fit(x = img_train, 
                        y = mask_train, 
                        validation_data = (img_test, mask_test), 
                        epochs = epochs, 
                        batch_size = 16,
                       callbacks = callbacks_list)
    end = time.time()
    delta = end - start
    print(str(timedelta(seconds=delta)))
    model.save("../logs/"+model_name+".h5")
    save_training(history, model_name)

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

def custom_loss(y_true, y_pred):
    cl =  bce_loss(y_true, y_pred) + dc_loss(y_true, y_pred) + idc_loss(y_true, y_pred)
    return cl

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

def CustomUNet(input_size=(256,256,1)):
    f = [16, 32, 64, 128, 256]
    inputs = tensorflow.keras.layers.Input(input_size)
    
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)

    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
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

def save_training(history, model_name):
    json_obj = json.dumps(str(history.history))
    f = open("../logs/"+model_name+"_history.json","w")
    f.write(json_obj)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom UNet Image Segmentation")
    parser.add_argument("--data_size", default=-1, type=int, help="No. of samples to use for Training and Testing combined. Use '-1' to use complete data")
    parser.add_argument("--img_size", default=256, type=int, help="Dimension of Images")
    parser.add_argument("--epochs", default=20, type=int, help="No. of Epochs")
    args = parser.parse_args()
    main(args)



