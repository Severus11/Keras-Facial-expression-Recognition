import numpy as np
import seaborn as sns
import utils 
import os

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten,Conv2D,BatchNormalization,Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from IPython.display import SVG, Image
from livelossplot.tf_keras import PlotLossesCallback

img_size=48
batch_size=64
datagen_train = ImageDataGenerator(horizontal_flip=True)
train_generator = datagen_train.flow_from_directory("Training_img/", target_size=(img_size, img_size), color_mode= 'grayscale', batch_size= batch_size, class_mode ='categorical', shuffle = True)
datagen_test = ImageDataGenerator(horizontal_flip=True)
test_generator = datagen_train.flow_from_directory("Test_img/", target_size=(img_size, img_size), color_mode= 'grayscale', batch_size= batch_size, class_mode ='categorical', shuffle = True)

model= Sequential()

model.add(Conv2D(64, (3,3), padding='same', input_size=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation='relu')
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation='relu')
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (5,5), padding= 'same'))
model.add(BatchNormalization())
model.add(Activation='relu')
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))

model.add(Conv2D(512, (3,3), padding='same'))

model.add(BatchNormalization())
model.add(Activation='relu')
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation='relu')
model.add(Dropout=0.25)

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation= 'relu')
model.add(Dropout=0.25)

model.add(Dense(7, activation='softmax'))

opt = Adam(lr=0.005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

nb_epoch = 20
steps= train_generator.n/ train_generator.batch_size
test_steps = test_generator.n/test_generator.batch_size

#callbacks to store the weights at the highest accuracy
checkpoint = ModelCheckpoint('model_weights.h5', monitor=['val_accuracy'], save_weights_only=True, mode='max', verbose=1)

#reduce the learning rate when the loss reaches a plateau condition
reduce_lr= ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience=2, min_delta=0.00001, mode='auto')

callbacks = [PlotLossesCallback(), checkpoint, reduce_lr]

history = model.fit(x= train_generator, steps_per_epoch=steps, epochs= nb_epoch, validation_data=test_generator, validation_steps=test_steps, callbacks=callbacks)
