# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:36:57 2017

@author: Toshiharu
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Conv2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping

from keras.regularizers import l2

EPOCHS = 10

model = Sequential()
model.add(Lambda(lambda x: (x/255-0.5)*2, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,25),(0,0))))  #Cropping top 60 pixels, bottom 25 pixels

# Layer 1
model.add(Conv2D(24,kernel_size=(5,5),activation='relu',subsample=(2,2)))  #(75-5+1)/2 = 36
#model.add(MaxPooling2D(pool_size=(2,2)))
# Layer 2
model.add(Conv2D(36,kernel_size=(5,5),activation='relu',subsample=(2,2)))  #(36-5+1)/2 = 16
# Layer 3
model.add(Conv2D(48,kernel_size=(5,5),activation='relu',subsample=(2,2)))   # (16-5+1)/2 = 6
# Layer 4
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',subsample=(2,2))) # (6-2+1)/1 = 5
# Layer 5
model.add(Conv2D(64,kernel_size=(2,2),activation='relu',subsample=(1,1))) # (6-2+1)/1 = 5
# Full dense layers
model.add(Flatten())
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='elu'))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='tanh'))
#model.add(Dense(1, W_regularizer = l2(0.001)))
#model.add(Dense(1, init='zero'))
#model.add(Dense(1, activation='elu',init='zero'))

adam = Adam(lr=1e-03, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.7) 
early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=1,
                                   min_delta=0.00009)

model.compile(loss='mse', optimizer = adam)

from keras.utils import plot_model
plot_model(model, to_file='model.png')

import pickle
import numpy as np

#training_file = 'G:/Documents/GITHUB/CarND-DataSets/data/driving_train3.p'
#with open(training_file, mode='rb') as f:
#    train = pickle.load(f)
#    
#X_train, y_train = train['features'], train['steering']
#
#import matplotlib.pyplot as plt
## Visualizations will be shown in the notebook.
#
#plt.hist(y_train,bins=50)
#plt.show()
#
#model.fit(X_train,y_train,validation_split = 0.2, shuffle=True, epochs=5,callbacks=[early_stopping])

#training_file = 'G:/Documents/GITHUB/CarND-Behavioral-Cloning-P3/track1cw.p'
#with open(training_file, mode='rb') as f:
#    train = pickle.load(f)
#    
#X_train, y_train = train['features'], train['steering']
#
#import matplotlib.pyplot as plt
## Visualizations will be shown in the notebook.
#
#plt.hist(y_train,bins=50)
#plt.show()
#
#model.fit(X_train,y_train,validation_split = 0.2, shuffle=True, epochs=EPOCHS,callbacks=[early_stopping])
#
#
training_file = 'G:/Documents/GITHUB/CarND-DataSets/data/track1full2.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
    
X_train, y_train = train['features'], train['steering']

import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.

plt.hist(y_train,bins=50)
plt.show()

model.fit(X_train,y_train,validation_split = 0.2, shuffle=True, epochs=EPOCHS,callbacks=[early_stopping])

model.save('track1full2.h5')
