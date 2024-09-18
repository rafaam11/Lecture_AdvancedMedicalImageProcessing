import numpy as np
import os
import SimpleITK as sitk
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

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
train_sol = np.empty((n_train, 1))
val_img = np.empty((n_val, ch, H, W))
val_sol = np.empty((n_val, 1))
test_img = np.empty((n_test, ch, H, W))
test_sol = np.empty((n_test, 1))

Data_path = os.getcwd()

# train 데이터 리스트 생성
walking_train = os.path.join(Data_path, 'clf_w_mask', 'train')
i = 0
for path, dirs, files in os.walk(walking_train):
    if 'img.npy' in files:
        path_img = os.path.join(path, 'img.npy')
        path_label = os.path.join(path, 'label.npy')
        img = np.load(path_img)
        label = np.load(path_label)
        train_img[i] = img
        train_sol[i] = label
        i += 1

# validation 데이터 리스트 생성
walking_val = os.path.join(Data_path,'clf_w_mask', 'valid')
i = 0
for path, dirs, files in os.walk(walking_val):
    if 'img.npy' in files:
        path_img = os.path.join(path, 'img.npy')
        path_label = os.path.join(path, 'label.npy')
        img = np.load(path_img)
        label = np.load(path_label)
        val_img[i] = img
        val_sol[i] = label
        i += 1

# test 데이터 리스트 생성
walking_test = os.path.join(Data_path, 'clf_w_mask', 'test')
i = 0
for path, dirs, files in os.walk(walking_test):
    if 'img.npy' in files:
        path_img = os.path.join(path, 'img.npy')
        path_label = os.path.join(path, 'label.npy')
        img = np.load(path_img)
        label = np.load(path_label)
        test_img[i] = img
        test_sol[i] = label
        i += 1

# img의 Shape를 조정 (n, 4, 240, 240) -> (n, 240, 240, 4)
train_img = np.transpose(train_img, (0, 2, 3, 1))
val_img = np.transpose(val_img, (0, 2, 3, 1))
test_img = np.transpose(test_img, (0, 2, 3, 1))

# img의 intensity를 0~1사이 범위로 mapping
print('Before train(min, max) :', np.min(train_img), np.max(train_img))
print('Before val(min, max) :', np.min(val_img), np.max(val_img))
print('Before test(min, max) :', np.min(test_img), np.max(test_img))
train_img = ( train_img - np.min(train_img) ) / (np.max(train_img) - np.min(train_img))
val_img = ( val_img - np.min(val_img) ) / (np.max(val_img) - np.min(val_img))
test_img = ( test_img - np.min(test_img) ) / (np.max(test_img) - np.min(test_img)) 
print('After train(min, max) :', np.min(train_img), np.max(train_img))
print('After val(min, max) :', np.min(val_img), np.max(val_img))
print('After test(min, max) :', np.min(test_img), np.max(test_img))

# ################################################################################
# CNN 모델 생성

# train model
model = Sequential()

# Convolution Layer 16
model.add(Conv2D(16, (3, 3), padding='same', input_shape=train_img.shape[1:]))
model.add(Activation('relu'))
# Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolution Layer 32
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
# Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully Connected Layer 512
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# OUTPUT
model.add(Dense(1))
model.add(Activation('sigmoid'))

# model compile
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['acc'])

# 학습 모델 출력
model.summary()

# ################################################################################
# train, val을 모델에 fitting

epochs = 32
hist = model.fit(train_img, train_sol,
            epochs=epochs,
            shuffle=True,
            validation_data=(val_img, val_sol),
            verbose=1)

# 모델 저장
learning_model_path = os.path.join(Data_path, 'HGGLGG_trained_model.h5')
model.save(learning_model_path)
print('Saved trained model at %s ' % learning_model_path)

# train accuray, loss 출력
train_acc = hist.history['acc'][-1]
train_loss =  hist.history['loss'][-1]
print('train accuracy: ', train_acc)
print('train loss: ', train_loss)


# train, val의 accuracy, loss를 그래프로 표현
plot_target = ['loss', 'val_loss', 'acc', 'val_acc']
plt.figure(figsize = (12,8))

plt.subplot(121)
plt.plot(hist.history['acc'], 'ro', label='Training accuracy')
plt.plot(hist.history['val_acc'], 'r', label='Validation accuracy')
plt.title('Trainging and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(122)
plt.plot(hist.history['loss'], 'bo', label='Training loss')
plt.plot(hist.history['val_loss'], 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.show()

# ################################################################################
# Test 모델로 성능 테스트

# predict 함수 정의
def predict_HGGLGG(x, model):
    x_data =(np.expand_dims(x, 0))
    predict = model.predict(x_data)
    #print(predict)
    if predict < 0.5:
        return 'LGG'
    else:
        return 'HGG'

# test 모델 84개에 대한 학습된 CNN의 예상 평가 결과
model = load_model(learning_model_path)
for p in range(n_test):
    test_data = test_img[p]
    result = predict_HGGLGG(test_data, model)
    print(p+1, '번째 이미지는', result, '입니다.')
