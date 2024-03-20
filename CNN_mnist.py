# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 21:10:52 2023

@author: Ellen Wang
"""


import tensorflow as tf
  
# conv_layer = tf.keras.layers.Conv2D(
#     filters, kernel_size, strides=(1, 1), padding='valid',
#     data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
#     use_bias=True, kernel_initializer='glorot_uniform',
#     bias_initializer='zeros', kernel_regularizer=None,
#     bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
#     bias_constraint=None, **kwargs
# )

# max_pooling_layer = tf.keras.layers.MaxPool2D(
#     pool_size=(2, 2), strides=None, padding='valid', data_format=None,
#     **kwargs
# )
  
# avg_pooling_layer = tf.keras.layers.AveragePooling2D(
#     pool_size=(2, 2), strides=None, padding='valid', data_format=None,
#     **kwargs
# )

# fully_connected_layer = tf.keras.layers.Dense(
#     units, activation=None, use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros', kernel_regularizer=None,
#     bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
#     bias_constraint=None, **kwargs
# )


from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')
  
# Plotting random images from dataset
  
import matplotlib.pyplot as plt 
import random
plt.figure(figsize = (12,5))
for i in range(8):
  ind = random.randint(0, len(X_train))
  plt.subplot(240+1+i)
  plt.imshow(X_train[ind])


from tensorflow.keras.utils import to_categorical
    
# convert image datatype from integers to floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
    
# normalising piel values
X_train = X_train/255.0
X_test = X_test/255.0
    
# reshape images to add channel dimension
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
# One-hot encoding label 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
  
model = Sequential()
  
# Layer 1
# Conv 1
model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=1, activation = 'relu', input_shape = (28,28,1)))
# Pooling 1
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
  
# Layer 2
# Conv 2
model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=1, activation='relu'))
# Pooling 2
model.add(MaxPooling2D(pool_size = 2, strides = 2))
  
# Flatten
model.add(Flatten())
   
# Layer 3
# Fully connected layer 1
model.add(Dense(units=120, activation='relu'))
  
#Layer 4
#Fully connected layer 2
model.add(Dense(units=84, activation='relu'))
  
#Layer 5
#Output Layer
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

epochs = 20
batch_size = 512
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                    steps_per_epoch=X_train.shape[0]//batch_size, 
                    validation_data=(X_test, y_test), 
                    validation_steps=X_test.shape[0]//batch_size, verbose = 1)

model.save("CNN_mnist_model1")
  
_, acc = model.evaluate(X_test, y_test, verbose = 1)
print('%.3f' % (acc * 100.0))
  
fig, ax = plt.subplots(1,2,figsize=(10,6))
ax[0].plot(history.epoch, history.history['accuracy'], color = 'blue', label = 'train')
ax[0].plot(history.epoch, history.history['val_accuracy'], color = 'red', label = 'val')
ax[0].legend()
ax[0].set_title('accuracy')

ax[1].plot(history.epoch, history.history['loss'], color = 'blue', label = 'train')
ax[1].plot(history.epoch, history.history['val_loss'], color = 'red', label = 'val')
ax[1].legend()
ax[1].set_title('loss')

# plt.show()

# load saved model
import keras
model1 = keras.models.load_model("CNN_mnist_model1")
_, acc1 = model1.evaluate(X_test, y_test, verbose = 1)
print('accuracy of loaded model = %.3f' % (acc1 * 100.0))






