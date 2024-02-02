import os
import math
import time
from datetime import timedelta
from collections import defaultdict
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import measure
import cv2
from tqdm import tqdm
import PIL
from glob import glob

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.activations import *
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

	model = unet(input_size=(img_size, img_size,1))
	model.compile(optimizer=Adam(lr=5*1e-4), loss="binary_crossentropy", metrics=[dice_coef, 'binary_accuracy'])
	print(model.summary())


	model_name = 'unet'
	weight_path="../logs/{}_weights.best.hdf5".format(model_name)

	checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True, save_weights_only = True)
	early = EarlyStopping(monitor="val_loss", patience=10)
	callbacks_list = [checkpoint, early]

	start = time.time()
	history = model.fit(x = img_train, 
	                    y = mask_train, 
	                    validation_data = (img_test, mask_test),
	                    epochs = epochs, 
	                    batch_size = 16,
	                   callbacks = callbacks_list,
	                   verbose = 2)
	end = time.time()
	delta = end - start
	print("Time taken for training:")
	print(str(timedelta(seconds=delta)))
	model.save('../logs/'+model_name+'.h5')
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

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
   
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])

def save_training(history, model_name):
    json_obj = json.dumps(str(history.history))
    f = open("../logs/"+model_name+"_history.json","w")
    f.write(json_obj)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNET Image Segmentation")
    parser.add_argument("--data_size", default=-1, type=int, help="No. of samples to use for Training and Testing combined. Use '-1' to use complete data")
    parser.add_argument("--img_size", default=256, type=int, help="Dimension of Images")
    parser.add_argument("--epochs", default=20, type=int, help="No. of Epochs")
    args = parser.parse_args()
    main(args)



