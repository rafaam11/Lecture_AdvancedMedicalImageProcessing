import tensorflow as tf
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

# 데이터를 준비합니다. 
(mnist_x, mnist_y), _ = tf.keras.datasets.mnist.load_data()
mnist_x = mnist_x.reshape(60000, 28, 28, 1)
mnist_y = pd.get_dummies(mnist_y)
print(mnist_x.shape, mnist_y.shape)

# 모델을 완성합니다. 
X = tf.keras.layers.Input(shape=[28, 28, 1])

#padding = 'same'은 특징맵의 사이즈를 입력 사이즈와 똑같게 출력하도록 해준다.
H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X) 
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

model.fit(mnist_x, mnist_y, epochs=10)

model.summary()

# cifar 10
# 데이터를 준비합니다. 
(cifar_x, cifar_y), _ = tf.keras.datasets.cifar10.load_data()
print(cifar_x.shape, cifar_y.shape)

cifar_y = pd.get_dummies(cifar_y.reshape(50000))
print(cifar_x.shape, cifar_y.shape)

# 모델을 완성합니다.
X = tf.keras.layers.Input(shape=[32, 32, 3])

H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

model.fit(cifar_x, cifar_y, epochs=10)

#BatchNormalization 적용
# 모델을 완성합니다.
X = tf.keras.layers.Input(shape=[32, 32, 3])

H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)

H = tf.keras.layers.Dense(120)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)

H = tf.keras.layers.Dense(84)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation('swish')(H)

Y = tf.keras.layers.Dense(10, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

model.fit(cifar_x, cifar_y, epochs=50)

pred = model.predict(cifar_x[0:5])
pd.DataFrame(pred).round(2)
 
plt.imshow(cifar_x[1].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()

cifar_y[0:5]