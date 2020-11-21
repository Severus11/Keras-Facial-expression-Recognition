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
