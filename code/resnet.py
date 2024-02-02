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

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="selu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)                     #using "selu" activation function
        self.main_layers = [                                                    #defining main layers
            keras.layers.Conv2D(filters, 3, strides=strides,padding="same"),    #Adding convolution layers
            keras.layers.BatchNormalization(),                                  #Adding normalization layer
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,padding="same"),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [                                                #defining skip layer
                keras.layers.Conv2D(filters, 1, strides=strides,padding="same"),
                keras.layers.BatchNormalization()]

    def call(self, inputs):                                                     #call function
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)                                      #activation function on skip layer


def ResNet():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(512, 7, strides=2, input_shape=np.shape(X_train[0]), padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("selu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(128, 3, strides=1, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("selu"))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same"))

    prev_filters = 128
    for filters in [128]*2 + [64]*2:                    #Adding Residual Unit
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=2))
        prev_filters = filters

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128,activation = 'selu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(64,activation = 'selu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(4, activation="softmax"))

    model.compile(loss='sparse_categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics=['accuracy'])

    return model

ResNetmodel = ResNet()
print('Summary of ResNet CNN Model:-')
print(ResNetmodel.summary())

checkpoint_path_resnet = '../logs/resnet.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path_resnet)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_resnet,
                                                save_weights_only=True,
                                                monitor = 'val_accuracy',
                                                mode = 'max',
                                                save_best_only=True,
                                                verbose=1)

ResNetmodel_hist = ResNetmodel.fit(X_train, y_train, epochs = 30, batch_size = 32, validation_split = 0.1, callbacks = [cp_callback])

def visualize_plots(ResNetmodel_hist):
    fig, axs = plt.subplots(1,2, figsize = (20,7))
    plt.title("ResNet Model")
    axs[0].plot(np.arange(1,31,1), ResNetmodel_hist.history['accuracy'], label = 'Accuracy')
    axs[0].plot(np.arange(1,31,1), ResNetmodel_hist.history['val_accuracy'], label = 'Val Accuracy')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(np.arange(1,31,1), ResNetmodel_hist.history['loss'], label = 'Loss')
    axs[1].plot(np.arange(1,31,1), ResNetmodel_hist.history['val_loss'], label = 'Val Loss')
    axs[1].legend()
    axs[1].grid()

    plt.show()

prob_pred_ResNet = ResNetmodel.predict(X_test)
y_pred_ResNet = np.argmax(prob_pred_ResNet, axis=1)
print('Accuracy on Test Set:',accuracy_score(y_test, y_pred_ResNet))
print('Classification Report:-')
print(classification_report(encoder.inverse_transform(y_test), encoder.inverse_transform(y_pred_ResNet)))


def ROC_Curve(y_test, prob_pred):
    binarizer = LabelBinarizer()
    y_test_bin = binarizer.fit_transform(y=y_test)
    y_score = prob_pred
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(4):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])


    mean_tpr /= 4  
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(14,12))
    for i in range(4):
        plt.plot(fpr[i],tpr[i],label="ROC curve of class {0} (area = {1:0.2f})".format(encoder.inverse_transform([i])[0], roc_auc[i]))

    plt.plot(fpr["micro"],tpr["micro"],label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),linestyle="--",)

    plt.plot(fpr["macro"],tpr["macro"],label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),linestyle="--",)

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

ROC_Curve(y_test, prob_pred_ResNet)
