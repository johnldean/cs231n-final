from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers import Dense, Activation
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
img_width, img_height = 256, 256
train_data_dir = "car_rocket/train"
validation_data_dir = "car_rocket/test"
nb_train_samples = 600
nb_validation_samples = 600
batch_size = 600
epochs = 3

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
N_tr = X_train.shape[0]
N_te = X_test.shape[0]
X_train = X_train.reshape(N_tr,-1)
X_test = X_train.reshape(N_te,-1)
in_size = X_train.shape[1]
#Adding custom Layers 
model = Sequential([
	Dense(1000,input_shape=(in_size,),kernel_regularizer=regularizers.l2(0.01)),
	Activation('relu'),
	Dropout(.5),
	Dense(100),
	Activation('relu'),
	Dropout(.5),
	Dense(3,kernel_regularizer=regularizers.l2(0.01)),
	Activation('softmax')
	])

model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model.fit(X_train,y_train,
	validation_data=(X_test,y_test),
	batch_size = 50,
	epochs=50)
