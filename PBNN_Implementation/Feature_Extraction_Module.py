## This code implements the self-supervised learning to extract representation features ##

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
import pandas as pd
import re
import scipy

from scipy.interpolate import interp1d

#%% SECTION TO RUN WITH GPU

# Choose GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU ID to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0";

Config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
Config.gpu_options.allow_growth=True

#%% Define Parameters

boxW = 0.25 + 0.0225*2
boxL = 0.25 + 0.0225*2
boxH = 1

def Ogden_func(x, w):
    
    p1 = 2*w[0]/w[3] * (x**(w[3]-1) - x**(-0.5*w[3]-1))
    p2 = 2*w[1]/w[4] * (x**(w[4]-1) - x**(-0.5*w[4]-1))
    p3 = 2*w[2]/w[5] * (x**(w[5]-1) - x**(-0.5*w[5]-1))
    
    return p1 + p2 + p3

#%% Load New data

label_set_0 = np.load('syntac_foam_new.npy')
label_set_1 = np.load('syntac_foam_new_1.npy')

label_set_new = np.concatenate([label_set_0,label_set_1], axis=0)

#%% Data Splitting

l_s, l_x, l_y = label_set_new.shape

X_train, X_test = train_test_split(label_set_new, test_size=0.4, random_state=47)
X_test,  X_cv   = train_test_split(X_test,        test_size=0.5, random_state=47)

#%% Define Convolutional Network Functions

def conv_relu_block(x,filt,kernel,stride,names):
    
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=kernel, strides=stride,
                               padding='same', activation='linear', 
                               use_bias=True,name=names)(x)
    y = tf.keras.layers.ReLU()(y)

    y = tf.keras.layers.BatchNormalization()(y)
    
    return y

def se_block(x,filt,ratio=16):
    
    init = x
    se_shape = (1, 1, filt)
    
    se = tf.keras.layers.GlobalAveragePooling2D()(init)
    se = tf.keras.layers.Reshape(se_shape)(se)
    se = tf.keras.layers.Dense(filt // ratio, activation='relu', 
                               kernel_initializer='he_normal', 
                               use_bias=False)(se)
    se = tf.keras.layers.Dense(filt, activation='sigmoid', 
                               kernel_initializer='he_normal', 
                               use_bias=False)(se)    
    se = tf.keras.layers.multiply([init, se])
    
    return se
    
def me_block(x,filt,ratio=16):
    
    init = x
    me_shape = (1, 1, filt)
    
    me = tf.keras.layers.GlobalMaxPooling2D()(init)
    me = tf.keras.layers.Reshape(me_shape)(me)
    me = tf.keras.layers.Dense(filt // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(me)
    me = tf.keras.layers.Dense(filt, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(me)
    
    me = tf.keras.layers.multiply([init, me])
    
    return me

def resnet_block(x,filt):

    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[3,3], 
                               padding='same', activation='linear', 
                               use_bias=True)(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[3,3], 
                               padding='same', activation='linear',
                               use_bias=True)(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = se_block(y,filt)
     
    y = tf.keras.layers.Add()([y,x])
    
    return y

def maxnet_block(x,filt):
    
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[3,3], 
                               padding='same', activation='linear', 
                               use_bias=True)(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[3,3], 
                               padding='same', activation='linear',
                               use_bias=True)(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = me_block(y,filt)
     
    y = tf.keras.layers.Add()([y,x])
    
    return y

def dense_block(x,filt,names):
    
    y = tf.keras.layers.Dense(filt, activation='relu', 
                              kernel_initializer='he_normal', use_bias=False,
                              name = names)(x)
    
    y = tf.keras.layers.BatchNormalization()(y)
    
    return y

#%% Setting up Convolutional Neural Network

input_layer = tf.keras.Input(shape=(l_x, l_y, 1))

conv_1 = conv_relu_block(input_layer, 32, kernel=[2,2], stride=2, names='conv1')

conv_2 = conv_relu_block(conv_1,  64, kernel=[4,4], stride=4, names='conv2')
conv_3 = conv_relu_block(conv_2,  64, kernel=[4,4], stride=4, names='conv3')
conv_4 = conv_relu_block(conv_3,  128, kernel=[4,4], stride=4, names='conv4')
conv_5 = conv_relu_block(conv_4,  128, kernel=[2,2], stride=2, names='conv5')

output_layer = conv_5

model1 = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

model1.summary()

input_layer1 = tf.keras.Input(shape=(1,1,128))

deconv_1 = tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=[2,2],
    strides=(2,2),padding='same',activation='linear', use_bias=True,
    name='deconv1')(input_layer1)
deconv_1 = tf.keras.layers.ReLU()(deconv_1)
# deconv_1 = tf.keras.layers.BatchNormalization()(deconv_1)

deconv_2 = tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=[4,4],
    strides=(4,4),padding='same',activation='linear', use_bias=True,
    name='deconv2')(deconv_1)
deconv_2 = tf.keras.layers.ReLU()(deconv_2)
# deconv_2 = tf.keras.layers.BatchNormalization()(deconv_2)

deconv_3 = tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=[4,4],
    strides=(4,4),padding='valid',activation='linear', use_bias=True,
    name='deconv3')(deconv_2)
deconv_3 = tf.keras.layers.ReLU()(deconv_3)

deconv_4 = tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=[4,4],
    strides=(4,4),padding='same',activation='linear', use_bias=True,
    name='deconv4')(deconv_3)
deconv_4 = tf.keras.layers.ReLU()(deconv_4)

deconv_5 = tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=[2,2],
    strides=(2,2),padding='valid',activation='linear', use_bias=True,
    name='deconv5')(deconv_4)
deconv_5 = tf.keras.layers.ReLU()(deconv_5)

output_layer1 = deconv_5

model2 = tf.keras.models.Model(inputs=input_layer1, outputs=output_layer1)

model2.summary()

#%% Setting up Convolutional Neural Network

def define_gan(g_model, d_model):
    
    d_model.trainable = True
    
    input_1 = tf.keras.Input(shape=(l_x, l_y, 1))

    inter_output = g_model(input_1)
    
    output = d_model(inter_output)
    
    gan_model = tf.keras.models.Model(inputs=input_1, outputs=output)
    
    return gan_model

gan_model = define_gan(model1, model2)

gan_model.summary()

#%% Training the model

import tensorflow.keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-6)
sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.6, nesterov=True)
# model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
gan_model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

epoch   = 500
history = gan_model.fit(X_train, X_train, batch_size=64, epochs=epoch, 
                    steps_per_epoch=40, validation_data=(X_cv, X_cv))

predict = gan_model.predict(X_test)

score = gan_model.evaluate(X_test, X_test, verbose=1)
print('\n', 'Test accuracy', score)

#%% Generating history plots of training

# Summarize history for accuracy
fig_acc = plt.figure()
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('model accuracy in training')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig('training_accuracy.png')

fig_acc_log = plt.figure()
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('model accuracy in training')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.yscale('log')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc_log.savefig('training_accuracy_log.png')

#%% Summarize history for loss
fig_loss_log = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.yscale('log')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_loss_log.savefig('training_loss_log.png')

fig_loss = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_loss.savefig('training_loss.png')

#%% Calculate the prediction error

plt.figure()
plt.imshow(X_test[0])
plt.colorbar()
plt.grid(True)
plt.title('Real Sample 1')

plt.figure()
plt.imshow(np.round(predict[0]))
plt.colorbar()
plt.grid(True)
plt.title('Pred Sample 1')

plt.figure()
plt.imshow(X_test[100])
plt.colorbar()
plt.grid(True)
plt.title('Real Sample 2')

plt.figure()
plt.imshow(np.round(predict[100]))
plt.colorbar()
plt.grid(True)
plt.title('Pred Sample 2')

plt.figure()
plt.imshow(X_test[500])
plt.colorbar()
plt.grid(True)
plt.title('Real Sample 3')

plt.figure()
plt.imshow(np.round(predict[500]))
plt.colorbar()
plt.grid(True)
plt.title('Pred Sample 3')

#%% Save the model

model1.save('conv_model_128_good_new.h5')
model2.save('deconv_model_128_good_new.h5')
