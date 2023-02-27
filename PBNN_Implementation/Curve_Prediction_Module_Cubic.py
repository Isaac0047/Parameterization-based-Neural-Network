## This code implements the code to represent  ##

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

boxW = 0.25 + 0.025*2
boxL = 0.25 + 0.025*2
boxH = 1

#%% Load stress-strain curve

strain_set = np.linspace(0,1,21) * 0.15
lambda_set = strain_set + 1

# ss_set = np.load('cubic_params.npy')
ss_set = np.load('cubic_params_update.npy')

# Thick1 -- Thick2 -- Thin1 -- Thin2
label_set = np.load('syntac_foam_new.npy')  

label_set = label_set[0:6925]

#%% Load back to original

ss_set[:,0] = ss_set[:,0] / 10000
ss_set[:,1] = ss_set[:,1] / 1000
ss_set[:,2] = ss_set[:,2] / 100

#%% 
plt.figure()
plt.plot(ss_set[:,0])
plt.title('Cubic Function Parameter $a_3$', fontsize=18)
plt.xlabel('Sample ID', fontsize=16)
plt.ylabel('Parameter values', fontsize=16)
plt.grid()
#plt.xticks(fontsize=14, rotation=90)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.figure()
plt.plot(ss_set[:,1])
plt.title('Cubic Function Parameter $a_2$', fontsize=18)
plt.xlabel('Sample ID', fontsize=16)
plt.ylabel('Parameter values', fontsize=16)
plt.grid()
#plt.xticks(fontsize=14, rotation=90)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.figure()
plt.plot(ss_set[:,2])
plt.title('Cubic Function Parameter $a_1$', fontsize=18)
plt.xlabel('Sample ID', fontsize=16)
plt.ylabel('Parameter values', fontsize=16)
plt.grid()
#plt.xticks(fontsize=14, rotation=90)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#%% Load existing model

cnn_model = tf.keras.models.load_model('conv_model_128_good_new.h5')
endpoint_model = tf.keras.models.load_model('trained_dense_end_point_128_new2.h5')

#%% Predefine Function

def Cubic_func(w):
    
    x=1.15
    
    p1 = w[0]*10000*(x-1)**3 + w[1]*1000*(x-1)**2 + w[2]*100*(x-1)**1 
    
    return p1

#%% Function Terms
[p1,p2] = ss_set.shape

slope_term = np.zeros((p1,1))

lam = 1.15

for i in range(p1):
    # slope_term[i][0] = ss_set[i][0] + ss_set[i][1] + ss_set[i][2]
    slope_term[i][0] = Cubic_func(ss_set[i,:])

#%% Concatenation
ss_set_slope = np.concatenate((ss_set,slope_term),axis=1)

#%% Combine the two inputs together

#%% Data Splitting

l_s, l_x, l_y = label_set.shape

X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(label_set, ss_set_slope, slope_term, test_size=0.4, random_state=47)
X_test,  X_cv,   Y_test,  Y_cv,   S_test,  S_cv   = train_test_split(X_test,    Y_test,       S_test,     test_size=0.5, random_state=47)

#%% Define Convolutional Network Functions

def dense_tanh_drop_block(x,filt,names):
    
    y = tf.keras.layers.Dense(filt, activation='linear', use_bias=True,
                              name=names)(x)
    
    y = tf.keras.activations.tanh(y)
    
    y = tf.keras.layers.Dropout(0.2)(y)
    
    return y

def dense_block(x,filt):
    
    y = tf.keras.layers.Dense(filt, activation='linear', use_bias=True)(x)
    
    return y

def dense_selu_block(x,filt,names):
    
    y = tf.keras.layers.Dense(filt, kernal_initializer='lecun_normal', 
                              activation='selu', use_bias=True,
                              name=names)(x)
    
    return y

def dense_relu_block(x,filt,names):
    
    y = tf.keras.layers.Dense(filt, activation='linear', use_bias=True,
                              name=names)(x)
    
    y = tf.keras.layers.ReLU()(y)

    return y

def dense_tanh_block(x,filt,names):
    
    y = tf.keras.layers.Dense(filt, activation='linear', use_bias=True,
                              name=names)(x)
    
    y = tf.keras.activations.tanh(y)
    
    #y = tf.keras.layers.Dropout(0.5)(y)
    
    return y

#%% Define the Neural Network

input_layer = tf.keras.Input(shape=(128))

dense_1 = dense_tanh_block(input_layer, 128, 'dense_01')
dense_2 = dense_tanh_block(dense_1,      64, 'dense_02')

output_layer = dense_block(dense_2, 4)

dense_model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

dense_model.summary()

#%% Setting up whole framework

def define_gan(g_model, d_model):
    
    g_model.trainable = False
    
    input_1 = tf.keras.Input(shape=(l_x, l_y, 1))

    inter_output = g_model(input_1)
    feature      = tf.reshape(inter_output, [-1,128])
    
    output    = d_model(feature)
    
    gan_model = tf.keras.models.Model(inputs=input_1, outputs=output)
    
    return gan_model

gan_model = define_gan(cnn_model, dense_model)

gan_model.summary()


#%% Train the model

import tensorflow.keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-6)
sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.6, nesterov=True)
gan_model.compile(optimizer=opt, loss='mean_squared_error', metrics='accuracy')
# gan_model.compile(optimizer=opt, loss=keras_custom_loss_function_test, metrics='accuracy')

epoch   = 350
history = gan_model.fit(X_train, Y_train, 
                    batch_size=128, epochs=epoch, steps_per_epoch=40, 
                    validation_data=(X_cv, Y_cv))

predict = gan_model.predict(X_test)

score = gan_model.evaluate(X_test, Y_test, verbose=1)
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


#%% Plot Prediction VS Truth

plt.figure()
plt.plot(Y_test[:,0])
plt.plot(predict[:,0])
plt.title('prediction of first parameter')
plt.legend(['true','predict'])

plt.figure()
plt.plot(Y_test[:,1])
plt.plot(predict[:,1])
plt.title('prediction of second parameter')
plt.legend(['true','predict'])

plt.figure()
plt.plot(Y_test[:,2])
plt.plot(predict[:,2])
plt.title('prediction of third parameter')
plt.legend(['true','predict'])

#%%
pred_mid0 = cnn_model(tf.reshape(X_test[0:800], [-1,256,256,1]))
pred_mid0 = tf.squeeze(pred_mid0)


#%%
pred_mid1 = cnn_model(tf.reshape(X_test[800:], [-1,256,256,1]))
pred_mid1 = tf.squeeze(pred_mid1)

#%%
pred_mid  = np.concatenate([pred_mid0,pred_mid1],axis=0)

#%%
pred_end = endpoint_model(pred_mid)

#%%
plt.figure()
plt.plot(Y_test[:,3])
plt.plot(predict[:,3])
plt.title('prediction of endpoint parameter')
plt.legend(['true','predict'])

#%% Visualize the result

art_cst = 0

def func_sca(x, m):
    
    p1 = 2*m[0]/(m[3]-art_cst) * (x**(m[3]-art_cst-1) - x**(-0.5*(m[3]-art_cst)-1))
    p2 = 2*m[1]/m[4] * (x**(m[4]-1) - x**(-0.5*m[4]-1))
    p3 = 2*m[2]/m[5] * (x**(m[5]-1) - x**(-0.5*m[5]-1))
    
    return p1 + p2 + p3

def func_sca_abs(x, m):
    
    m0 = m[0]*10000
    m1 = m[1]*1000
    m2 = m[2]*100
    
    p1 = m0*(x-1)**3 + m1*(x-1)**2 + m2*(x-1)**1
    
    return p1

# Take test set 1

test_index = 0

stress_pred = func_sca_abs(lambda_set, predict[test_index])
stress_real = func_sca_abs(lambda_set, Y_test[test_index])

plt.figure()
plt.plot(strain_set, stress_pred, 'r-')
plt.plot(strain_set, stress_real, 'b-')
plt.legend(['Predicted Curve','True Curve'])
plt.title('Validation on Test set 1')

# Take test set 2

test_index = 20

stress_pred = func_sca_abs(lambda_set, predict[test_index])
stress_real = func_sca_abs(lambda_set, Y_test[test_index])

plt.figure()
plt.plot(strain_set, stress_pred, 'r-')
plt.plot(strain_set, stress_real, 'b-')
plt.legend(['Predicted Curve','True Curve'])
plt.title('Validation on Test set 2')

# Take test set 3

test_index = 200

stress_pred = func_sca_abs(lambda_set, predict[test_index])
stress_real = func_sca_abs(lambda_set, Y_test[test_index])

plt.figure()
plt.plot(strain_set, stress_pred, 'r-')
plt.plot(strain_set, stress_real, 'b-')
plt.legend(['Predicted Curve','True Curve'])
plt.title('Validation on Test set 3')

#%% Evaluate the Norm Error Based on Curve

[p1, p2] = predict.shape

error1 = np.zeros((p1))
error2 = np.zeros((p1))
error3 = np.zeros((p1))

for i in range(p1):
    error1[i] = np.abs(predict[i][0] - Y_test[i][0])
    error2[i] = np.abs(predict[i][1] - Y_test[i][1])
    error3[i] = np.abs(predict[i][2] - Y_test[i][2])
    
error1_ave = np.mean(error1)
error2_ave = np.mean(error2)
error3_ave = np.mean(error3)

print('error_1_ave ', error1_ave)
print('error_2_ave ', error2_ave)
print('error_3_ave ', error3_ave)

#%% Evaluate the Norm Error Based on Curve

[p1,p2] = predict.shape

error = np.zeros((p1))
error_max = np.zeros((p1))

for i in range(p1):
    
    stress_pred  = func_sca_abs(lambda_set, predict[i,:])
    stress_real  = func_sca_abs(lambda_set, Y_test[i,:])
    error[i]     = np.linalg.norm(stress_pred-stress_real)
    error_max[i] = np.max(np.abs(stress_pred-stress_real))
       
print("Norm-2 Error based on ss curve is:", np.mean(error))
print("Norm-2 Error based on ss curve is:", np.max(error))

print("Absolute Error based on ss curve is:", np.mean(error_max))
print("Absolute Error based on ss curve is:", np.max(error_max))

#%%

gan_model.save('trained_cubic_128_final.h5')

#%% Evaluate the Error of Decoder Network Again

# endpoint_model = tf.keras.models.load_model('trained_dense_end_point_128_cubic.h5')
# endpoint_model = tf.keras.models.load_model('trained_end_point_128.h5')

#%%
pred_mid = cnn_model(X_test)
pred_mid = tf.squeeze(pred_mid)

#%%
pred_end = endpoint_model(pred_mid)

#%% Let's try a second modification method

# Here lambda is the independent variable

clip = 1.00

def quadratic(x, w):
    
    # clip = 1.08
    p  = (x>clip)*w*(x-clip)**2
    
    return p

def Ogden_func_end(x, w):
    
    m0 = w[0]*10000
    m1 = w[1]*1000
    m2 = w[2]*100
    
    p1 = m0*(x-1)**3 + m1*(x-1)**2 + m2*(x-1)**1
    
    return p1

#%% Let's implement second modification method

predict_new = np.zeros((p1,p2))
q1 = len(lambda_set)

end_value = np.zeros((p1,1)) 
delta_y   = np.zeros((p1,1))
curve_gap = np.zeros((p1,q1))

for i in range(p1):
    
    end_value[i] = Ogden_func_end(1.15,predict[i][0:3])
    
    delta_y[i]   = (end_value[i] - predict[i][3]) / ((1.15-clip)**2)
    # curve_gap[i,:]    = quadratic(lambda_set, delta_y)
    
#%% Now Re-evaluate the modified predictions

[p1,q1] = Y_test.shape

error_new_1 = np.zeros((p1))
error_new_2 = np.zeros((p1))

for i in range(p1):
    
    stress_pred      = Ogden_func_end(lambda_set, predict[i])
    stress_pred_new  = Ogden_func_end(lambda_set, predict[i]) - quadratic(lambda_set, delta_y[i])
    stress_real      = Ogden_func_end(lambda_set, Y_test[i])
    error_new_1[i]   = np.linalg.norm(stress_pred-stress_real)
    error_new_2[i]   = np.linalg.norm(stress_pred_new-stress_real)   
    
print("Norm-2 Error based on simple ss curve is:", np.mean(error_new_1))
print("Norm-2 Error based on simple ss curve is:", np.max(error_new_1))

print("Norm-2 Error based on modified ss curve is:", np.mean(error_new_2))
print("Norm-2 Error based on modified ss curve is:", np.max(error_new_2))

#%%

plt.figure()
plt.plot(error_new_1,'r-')
plt.plot(error_new_2,'b-')

#%%
plt.figure()
plt.plot(end_value,'r-')
plt.plot(predict[:,3],'b-')
plt.plot(Y_test[:,3], 'g-')

plt.title('Prediction of different curves')
plt.legend(['param_end_value','predict_end','true_end'])

#%% Analyze why the error is out

error_diff = error_new_2 - error_new_1
a = [x for x in range(len(error_diff)) if error_diff[x]>0]

#%%

# Take test set 1

test_index = 0

stress_pred     = Ogden_func_end(lambda_set, predict[test_index])
stress_pred_new = Ogden_func_end(lambda_set, predict[test_index]) - quadratic(lambda_set, delta_y[test_index])
stress_real     = Ogden_func_end(lambda_set, Y_test[test_index])

plt.figure()
plt.plot(strain_set, stress_pred, 'g*')
plt.plot(strain_set, stress_pred_new, 'r-')
plt.plot(strain_set, stress_real, 'b-')
plt.legend(['Predicted Curve','Predicted Modified Curve','True Curve'])
plt.title('Validation on Test set 1')

#%% Take test set 2

test_index = 20

stress_pred = Ogden_func_end(lambda_set, predict[test_index])
stress_pred_new = Ogden_func_end(lambda_set, predict[test_index]) - quadratic(lambda_set, delta_y[test_index])
stress_real = Ogden_func_end(lambda_set, Y_test[test_index])

plt.figure()
plt.plot(strain_set, stress_pred, 'g*')
plt.plot(strain_set, stress_pred_new, 'r-')
plt.plot(strain_set, stress_real, 'b-')
plt.legend(['Predicted Curve','Predicted Modified Curve','True Curve'])
plt.title('Validation on Test set 2')

#%% Take test set 3

test_index = 200

stress_pred = Ogden_func_end(lambda_set, predict[test_index])
stress_pred_new = Ogden_func_end(lambda_set, predict[test_index]) - quadratic(lambda_set, delta_y[test_index])
stress_real = Ogden_func_end(lambda_set, Y_test[test_index])

plt.figure()
plt.plot(strain_set, stress_pred, 'g*')
plt.plot(strain_set, stress_pred_new, 'r-')
plt.plot(strain_set, stress_real, 'b-')
plt.legend(['Predicted Curve','Predicted Modified Curve','True Curve'])
plt.title('Validation on Test set 3')
plt.xlabel('Strain')
plt.ylabel('Stress')

#%% Plot the coefficients

plt.figure()
plt.plot(ss_set)
plt.title('Ogden paramter values for different models',fontsize=14)
plt.xlabel('model#',fontsize=14)
plt.ylabel('parameter value',fontsize=14)
plt.legend([r'$\mu_1$',r'$\mu_2$',r'$\mu_3$',r'$\alpha_1$',r'$\alpha_2$',r'$\alpha_3$'])
