from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers import Dense, Activation
from keras import regularizers
from keras.losses import categorical_hinge
import tensorflow as tf
import keras as ke
import matplotlib.pyplot as plt
import numpy as np
import keras
import cvxpy as cp

sess = tf.InteractiveSession()
k.set_session(sess)

load_dir = "assets/car-rocket-ship/imnet_nofc/VGG19/128x128/"
model_dir = "assets/models/"
X_train = np.load(load_dir + 'X_train.npy')
y_train = np.load(load_dir + 'y_train.npy')
X_test = np.load(load_dir + 'X_test.npy')
y_test = np.load(load_dir + 'y_test.npy')
N_tr = X_train.shape[0]
N_te = X_test.shape[0]
X_train = X_train.reshape(N_tr,-1)
X_test = X_test.reshape(N_te,-1)
in_size = X_train.shape[1]
print("d")
def weights_grad(X,model):
    global sess
    N = X.shape[0]
    M = model.output.shape[1]
    w = model.trainable_weights
    out = model.layers[-1].output 
    grads = []
    for i in range(N): #iterate over the batch
        for j in range(M): #iterate over class score
            grads.append(k.gradients(out[i,j],w))
    grads_evaled,scores,weights = sess.run((grads,out,w),feed_dict={model.input:X})
    # grads_evaled is is an M*N list of the gradients of the weights.
    # each element of this list in another list, that will be something like
    # [W1_grad, b1_grad, W2_grad, b2_grad ...]
    return grads_evaled, scores, weights

reg = 0.01
random_model = Sequential([
        Dense(50,input_shape=(in_size,),kernel_regularizer=regularizers.l2(reg)),
        Activation('relu'),
        #Dropout(0.3),
        Dense(10,kernel_regularizer=regularizers.l2(reg)),
        Activation('relu'),
        #Dropout(0.3),
        Dense(3,kernel_regularizer=regularizers.l2(reg)),
        #Activation('softmax')
        ])

model = keras.models.load_model(model_dir + "tfnet_crs_50-10-3")
g,y,_ = weights_grad(X_train[0:2,:],model)
#print(y)
#print(len(g))

 
for layer in model.layers:
    print(layer)

model.layers[0].trainable = False
model.layers[2].set_weights(random_model.layers[2].get_weights())
model.layers[4].set_weights(random_model.layers[4].get_weights())
model.compile(loss = "categorical_hinge", optimizer = optimizers.SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])

g,y,_ = weights_grad(X_train[0:2,:],model)
#print(y)
#print(len(g)) 
batch_size = 20
num_weights = 50*10+10+10*3+3
num_classes = 3
w1_shape = (50,10)
b1_shape = (10)
w2_shape = (10,3)
b2_shape = (3)
indices = np.random.choice(range(N_tr), batch_size, replace=True)
for i in range(10):
    print("PREPAIRING")
    ytrain_inds = y_train.argmax(axis=1)
    X_batch = X_train[indices]
    true_class = ytrain_inds[indices]
    grads, scores, weights = weights_grad(X_batch,model)
    gstack = np.zeros((num_weights,num_classes*batch_size))
    for nn in range(batch_size):
        for class_ind in range(num_classes):
            gstack[:,[nn*num_classes + class_ind]] = np.vstack([param.reshape(-1,1) for param in (grads[3*nn + class_ind])])
    #print(gstack.shape)    

    y = scores

    keras_l =  np.sum(k.eval(categorical_hinge(y_train[indices,:],y)))
    #print(y)
    #print(y.reshape((-1,1)))
    wk = np.vstack([param.reshape(-1,1) for param in weights])

    print("SETTING UP CONVEX PROBLEM")
    w = cp.Variable((num_weights,1))
    #yhat = cp.Variable((batch_size,num_classes))
    #const = [yhat == y + cp.reshape((gstack.T@(w - wk)),(batch_size,num_classes))]
    const = []
    yhat = y + cp.reshape((gstack.T@(w - wk)),(batch_size,num_classes))
    #f = cp.sum(-yhat[np.arange(batch_size),true_class] + cp.log_sum_exp(yhat, axis=1)) + cp.norm(w,2)
    f = cp.sum(cp.pos(yhat - yhat[np.arange(batch_size),[true_class]].T@np.ones((1,3)) + 1)) + 10*cp.norm(w-wk,2)
    objective = cp.Minimize(f)
    prob = cp.Problem(objective, const)
    print("LEGGO")
    r = prob.solve(solver="SCS",verbose=False)
    print("Current Loss: ",keras_l)
    print("SOLVED!, l = ",r)
    w_new = w.value
    shapes=grads[0]
    ind = 0
    w_ = []
    #print(w_new.shape)
    for sh in shapes:
        w_.append(w_new[ind:ind+sh.size].reshape(sh.shape))
        ind += sh.size
    #print(w_[2][0,0],model.layers[4].get_weights()[0][0,0])
    model.layers[2].set_weights(w_[0:2])
    model.layers[4].set_weights(w_[2:4])
    model.compile(loss = "categorical_hinge", optimizer = optimizers.SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])


'''
hist = model.fit(X_train,y_train,
    validation_data=(X_test,y_test),
    batch_size = 100,
    epochs=25,
    verbose=2
    )

for layer in model.layers:
    print(layer.trainable)

plt.plot(hist.history['acc'], label="train")
plt.plot(hist.history['val_acc'], label="val")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
'''