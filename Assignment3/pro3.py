import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold
import tensorflow as tf
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


# img의 intensity를 0~1사이 범위로 Normarlize
for i in range(n_train):
    train_img[i] = ( train_img[i] - np.min(train_img[i]) ) / (np.max(train_img[i]) - np.min(train_img[i]))
    train_sol[i] = ( train_sol[i] - np.min(train_sol[i]) ) / (np.max(train_sol[i]) - np.min(train_sol[i]))

for i in range(n_val):
    val_img[i] = ( val_img[i] - np.min(val_img[i]) ) / (np.max(val_img[i]) - np.min(val_img[i]))
    val_sol[i] = ( val_sol[i] - np.min(val_sol[i]) ) / (np.max(val_sol[i]) - np.min(val_sol[i]))

for i in range(n_test):
    test_img[i] = ( test_img[i] - np.min(test_img[i]) ) / (np.max(test_img[i]) - np.min(test_img[i]))
    test_sol[i] = ( test_sol[i] - np.min(test_sol[i]) ) / (np.max(test_sol[i]) - np.min(test_sol[i])) 


# img의 Shape를 조정 (n, 4, 240, 240) -> (n, 240, 240, 4)
train_img = np.transpose(train_img, (0, 2, 3, 1))
val_img = np.transpose(val_img, (0, 2, 3, 1))
test_img = np.transpose(test_img, (0, 2, 3, 1))

train_x = train_img
valid_x = val_img
test_x = test_img

train_y = train_sol.reshape(n_train, 240, 240, 1)
valid_y = val_sol.reshape(n_val, 240, 240, 1)
test_y = test_sol.reshape(n_test, 240, 240, 1)

print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

# 모델 생성
def contract_path(input_shape):
    input= tf.keras.layers.Input(shape = input_shape)
    x =  Conv2D(64, (3,3), padding = "same", activation = "relu")(input)
    x =  Conv2D(64, (3,3), padding = "same", activation = "relu", name = "copy_crop1")(x)
    x = MaxPooling2D((2, 2))(x)
    x =  Conv2D(128, (3,3), padding = "same", activation = "relu")(x)
    x =  Conv2D(128, (3,3), padding = "same", activation = "relu", name = "copy_crop2")(x)
    x = MaxPooling2D((2, 2))(x)
    x =  Conv2D(256, (3,3), padding = "same", activation = "relu")(x)
    x =  Conv2D(256, (3,3), padding = "same", activation = "relu", name = "copy_crop3")(x)
    x = MaxPooling2D((2, 2))(x)
    x =  Conv2D(512, (3,3), padding = "same", activation = "relu")(x)
    x =  Conv2D(512, (3,3), padding = "same", activation = "relu", name = "copy_crop4")(x)
    x = MaxPooling2D((2, 2))(x)
    x =  Conv2D(1024, (3,3), padding = "same", activation = "relu")(x)
    x =  Conv2D(1024, (3,3), padding = "same", activation = "relu", name = "last_layer")(x)
    contract_path =  tf.keras.Model(inputs = input, outputs = x)
    return contract_path

def unet(input_shape, n_classes):
    contract_model = contract_path(input_shape=input_shape)
    layer_names  = ["copy_crop1", "copy_crop2",  "copy_crop3" ,"copy_crop4", "last_layer"]
    layers = [contract_model.get_layer(name).output for name in layer_names]

    extract_model = tf.keras.Model(inputs=contract_model.input, outputs=layers)
    input= tf.keras.layers.Input(shape =input_shape)
    output_layers = extract_model(inputs = input)
    last_layer = output_layers[-1]

    x = Conv2DTranspose(512, 4, (2,2), padding = "same", activation = "relu")(last_layer)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[3]])

    x =  Conv2D(256, (3,3), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)

    x =  Conv2D(256, (3,3), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(256, 4, (2,2), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[2]])

    x =  Conv2D(128, (3,3), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x =  Conv2D(128, (3,3), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(128, 4, (2,2), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[1]])


    x =  Conv2D(64, (3,3), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x =  Conv2D(64, (3,3), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, 4, (2,2), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[0]])

    x =  Conv2D(64, (3,3), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x =  Conv2D(64, (3,3), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x =  Conv2D(n_classes, (1,1), activation = "relu")(x)

    model = tf.keras.Model(inputs = input , outputs = x)

    return model


model = unet((240, 240, 4), 4)
model.summary()
model.compile(optimizer = 'adam', 
            loss = 'binary_crossentropy', 
            metrics = ['acc'])

hist = model.fit(train_x, train_y, 
                shuffle=True,
                validation_data=(valid_x, valid_y), 
                epochs = 24, 
                verbose = 1 )

# 모델 저장
learning_model_path = os.path.join(Data_path, 'U-NET_Segmentation_trained_model.h5')
model.save(learning_model_path)
print('Saved trained model at %s ' % learning_model_path)

# train val의 loss 출력
train_loss =  hist.history['loss'][-1]
val_loss =  hist.history['val_loss'][-1]
print('Training loss: ', train_loss)
print('validation loss: ', val_loss)

# train, val의 loss를 그래프로 표현
plt.figure(figsize = (12,8))
plt.plot(hist.history['loss'], 'bo', label='Training loss')
plt.plot(hist.history['val_loss'], 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# ################################################################################
# Test 모델 predict
my_model = load_model(learning_model_path)
test_img_number = 1
test_img_ch = 1
test_image = test_x[test_img_number]
ground_truth= test_y[test_img_number]
test_image_input=np.expand_dims(test_image, 0)
prediction = (my_model.predict(test_image_input)[0,:,:,0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(241)
plt.title('Testing Image')
plt.imshow(test_image[:,:,test_img_ch], cmap='gray')
plt.subplot(242)
plt.title('Testing Solution')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(243)
plt.title('Prediction')
plt.imshow(prediction, cmap='gray')
plt.subplot(244)
plt.title('Prediction on test image')
plt.imshow(test_image[:,:,test_img_ch]+prediction, cmap='gray')
plt.show()