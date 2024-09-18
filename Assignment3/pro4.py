import numpy as np
import os
from matplotlib import pyplot as plt
import glob
import shutil
import random
from PIL import Image
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold
from tensorflow.keras.layers import Dense, Input, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization,Add,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LeakyReLU, ReLU, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, UpSampling2D, concatenate
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K

# ################################################################################
# 데이터 셋업

# Global values 및 ndarray 생성
n_train = 222
n_val = 29
n_test = 84
ch = 4
H = 240
W = 240
train_img = np.empty((n_train, ch, H, W))
train_sol = np.empty((n_train, H, W))
val_img = np.empty((n_val, ch, H, W))
val_sol = np.empty((n_val, H, W))
test_img = np.empty((n_test, ch, H, W))
test_sol = np.empty((n_test, H, W))

Data_path = os.getcwd()

# train 데이터 리스트 생성
walking_train = os.path.join(Data_path, 'clf_w_mask', 'train')

i = 0
for path, dirs, files in os.walk(walking_train):
    if 'img.npy' in files:
        path_img = os.path.join(path, 'img.npy')
        path_seg = os.path.join(path, 'seg.npy')
        img = np.load(path_img)
        seg = np.load(path_seg)
        train_img[i] = img
        train_sol[i] = seg
        i += 1

# validation 데이터 리스트 생성
walking_val = os.path.join(Data_path,'clf_w_mask', 'valid')
i = 0
for path, dirs, files in os.walk(walking_val):
    if 'img.npy' in files:
        path_img = os.path.join(path, 'img.npy')
        path_seg = os.path.join(path, 'seg.npy')
        img = np.load(path_img)
        seg = np.load(path_seg)
        val_img[i] = img
        val_sol[i] = seg
        i += 1

# test 데이터 리스트 생성
walking_test = os.path.join(Data_path, 'clf_w_mask', 'test')
i = 0
for path, dirs, files in os.walk(walking_test):
    if 'img.npy' in files:
        path_img = os.path.join(path, 'img.npy')
        
        path_seg = os.path.join(path, 'seg.npy')
        img = np.load(path_img)
        seg = np.load(path_seg)
        test_img[i] = img
        test_sol[i] = seg
        i += 1


# img의 Shape를 조정 (n, 4, 240, 240) -> (n, 240, 240, 4)
train_img = np.transpose(train_img, (0, 2, 3, 1))
val_img = np.transpose(val_img, (0, 2, 3, 1))
test_img = np.transpose(test_img, (0, 2, 3, 1))

train_x = train_img
valid_x = val_img

train_y = train_sol.reshape(n_train, 240, 240, 1)
valid_y = val_sol.reshape(n_val, 240, 240, 1)

print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)


def UNet(pretrained_weights = None,input_size = (240, 240, 4)):
    inp = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inp)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inp, outputs=[conv10])

    return model

model = UNet()
model.summary()
model.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy', 
            metrics = ['acc'])

history = model.fit(train_x, train_y, 
                shuffle=True,
                validation_data=(valid_x, valid_y), 
                epochs = 5, 
                verbose = 1 )