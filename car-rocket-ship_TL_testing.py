from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers import Dense, Activation
from keras import regularizers
import tensorflow as tf
import keras as ke
import matplotlib.pyplot as plt
import numpy as np


load_dir = "assets/car-rocket-ship/imnet_nofc/VGG19/128x128/"
X_train = np.load(load_dir + 'X_train.npy')
y_train = np.load(load_dir + 'y_train.npy')
X_test = np.load(load_dir + 'X_test.npy')
y_test = np.load(load_dir + 'y_test.npy')
N_tr = X_train.shape[0]
N_te = X_test.shape[0]
X_train = X_train.reshape(N_tr,-1)
X_test = X_test.reshape(N_te,-1)
in_size = X_train.shape[1]
#choosing bewtween different network architectures  
reg=0.001
if 0:
    model = Sequential([
        Dense(100,input_shape=(in_size,),kernel_regularizer=regularizers.l2(reg)),
        Activation('relu'),
        Dropout(0.3),
        Dense(50,kernel_regularizer=regularizers.l2(reg)),
        Activation('relu'),
        Dropout(0.3),
        Dense(3,kernel_regularizer=regularizers.l2(reg)),
        Activation('softmax')
        ])
if 0:
    model = Sequential([
        Dense(3,input_shape=(in_size,),kernel_regularizer=regularizers.l2(0.001)),
        Activation('softmax')
        ])

if 1:
    model = Sequential([
        #Dropout(0,input_shape=(in_size,)),
        Dense(10,input_shape=(in_size,),kernel_regularizer=regularizers.l2(0.5)),
        Activation('relu'),
        Dense(3,kernel_regularizer=regularizers.l2(reg)),
        Activation('softmax')
        ])

model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])

def weights_grad(X,model):
	N = X.shape[0]
	M = model.output.shape[1]
	w = model.trainable_weights
	out = model.layers[-2].output 
	grads = []
	for i in range(N): #iterate over the batch
		for j in range(M): #iterate over class score
			grads.append(k.gradients(out[i,j],w))
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	grads_evaled = sess.run(grads,feed_dict={model.input:X})
	# grads_evaled is is an M*N list of the gradients of the weights.
	# each element of this list in another list, that will be something like
	# [W1_grad, b1_grad, W2_grad, b2_grad ...]
	return grads_evaled

# Train the model and plot training info

hist = model.fit(X_train,y_train,
    validation_data=(X_test,y_test),
    batch_size = 100,
    epochs=15,
    verbose=2
    )
plt.plot(hist.history['acc'], label="train")
plt.plot(hist.history['val_acc'], label="val")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()