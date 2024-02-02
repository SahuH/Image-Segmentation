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
import seaborn
import tensorflow as tf
from skimage import measure
import cv2
from tqdm import tqdm
import PIL
from glob import glob

from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

path_DIR ='../COVID-19_Radiography_Dataset'
train_csv = pd.read_csv('../saved_files/train_ids.csv', usecols=['train_ids'])
test_csv = pd.read_csv('../saved_files/test_ids.csv', usecols=['test_ids'])

test_ids = test_csv.test_ids.values.tolist()
train_ids = train_csv.train_ids.values.tolist()
test_df = []
train_df = []
for test_id in test_ids:
    if test_id.find("COVID")!=-1:
        path = path_DIR+'/COVID/images/'+test_id+'.png'
        test_df.append(path)
    if test_id.find("Normal")!=-1:
        path = path_DIR+'/Normal/images/'+test_id+'.png'
        test_df.append(path)

for train_id in train_ids:
    if train_id.find("COVID")!=-1:
        path = path_DIR+'/COVID/images/'+train_id+'.png'
        train_df.append(path)
    if train_id.find("Normal")!=-1:
        path = path_DIR+'/Normal/images/'+train_id+'.png'
        train_df.append(path)


def visualize_data(path_df):
    labels = ['COVID','Normal']
    for i in range(2):
        fig, axs = plt.subplots(1,5, figsize = (50,10))
        print('X-Ray Images of Class',labels[i],':-')
        paths = [k for k in path_df if(labels[i] in k)]
        for j in range(5):
            img = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(paths[j], color_mode = 'grayscale'))
            axs[j].imshow(img[:,:,0], cmap='gray')
        plt.show()

def remove_zero_pad(image):
    dummy = np.argwhere(image != 0) 
    max_y = dummy[:, 0].max()
    min_y = dummy[:, 0].min()
    min_x = dummy[:, 1].min()
    max_x = dummy[:, 1].max()
    crop_image = image[min_y:max_y, min_x:max_x]

    return crop_image

X_train = []
X_test = []
y_train = []
y_test = []
for i in train_df:
    img = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(i, color_mode = 'grayscale'))
    crop_img = remove_zero_pad(img)
    img = tf.image.resize(crop_img, (175, 175))
    X_train.append(np.array(img/255.0, dtype = np.float16))
    y_train.append(i.split('/')[-3])

for i in test_df:
    img = keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(i, color_mode = 'grayscale'))
    crop_img = remove_zero_pad(img)
    img = tf.image.resize(crop_img, (175, 175))
    X_test.append(np.array(img/255.0, dtype = np.float16))
    y_test.append(i.split('/')[-3])

print('Training images count:', len(y_train))
print('Test images count:', len(y_test))

def visualize_class_cnt(y_train, y_val, y_test):
    value_count_df = pd.DataFrame(pd.Series(np.concatenate([y_train, y_val, y_test])).value_counts()).rename_axis('unique_values').reset_index()
    value_count_df.columns = ['Class','Count']
    plt.bar(value_count_df['Class'], value_count_df['Count'])
    plt.show()

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

X_train = np.array(X_train)
X_test = np.array(X_test)

def customCNN(input_size=(img_size, img_size, 1)):
	model = Sequential([
	  layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_size),
	  layers.Conv2D(64, (3,3), padding='same', activation='relu'),
	  layers.MaxPooling2D(),
	  layers.Dropout(0.25),
	    
	  layers.Conv2D(64, (3,3), padding='same', activation='relu'),
	  layers.MaxPooling2D(),
	  layers.Dropout(0.25),
	    
	  layers.Conv2D(128, (3,3), padding='same', activation='relu'),
	  layers.MaxPooling2D(),
	  layers.Dropout(0.25),
	    
	  layers.Flatten(),
	  layers.Dense(128, activation='relu'),
	  layers.Dropout(0.5),
	  layers.Dense(1, activation='sigmoid')
	])

	return model

model = customCNN((img_size, img_size, 1))
print('Summary of Custom CNN Model:-')

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


checkpoint_filepath = './'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_stopping_callbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


print(model.summary())

# checkpoint_path_resnet = '../logs/customCNN.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path_resnet)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_resnet,
#                                                 save_weights_only=True,
#                                                 monitor = 'val_accuracy',
#                                                 mode = 'max',
#                                                 save_best_only=True,
#                                                 verbose=1)

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[model_checkpoint_callback, early_stopping_callbacks])


