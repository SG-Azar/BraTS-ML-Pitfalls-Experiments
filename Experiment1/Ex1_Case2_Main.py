
"""
Created on Mon NOV 4 11:34:05 2023


Ex1-new version


#######################     ONLY CASE2  - Revised    ######################################################



Train and extract results:

############################################################################################################
            tr                            val                           ts - a            ts - b
            
Case 1:     site1(id0)                   site1(id0)                   site1(id0)       site18 (id1)            
Case 2:       site 1,4,13,21,6,20 (id0,2,3,4,5,6)                     site1(id0)       site18 (id1)
############################################################################################################
    

@author: sgazar
"""
#%% Cell 0
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


#%% Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import math
import time
import numpy as np
import tensorflow as tf


import tensorflow.keras as keras
from tensorflow.keras.models import Model
from keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Dropout,\
                                    Add, AveragePooling2D, Flatten, Dense, UpSampling2D,\
                                    MaxPooling2D, GlobalAveragePooling2D, Activation, concatenate, Conv2DTranspose

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import binary_crossentropy


import matplotlib.pyplot as plt
from medpy.metric.binary import hd95, dc, hd, ravd, asd
from scipy import stats
import pandas as pd

#%%
plt.close('all')

#%% seed
seed = 10

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

tf.keras.utils.set_random_seed(seed)

#%% data generator

class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 data_path,
                 arrays,    # array labels
                 batch_size=32,
                 ):

        self.data_path = data_path
        self.arrays = arrays
        self.batch_size = batch_size

        if data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(data_path):
            raise ValueError('The data path is incorrectly defined.')

        self.file_idx = 0
        self.file_list = [self.data_path + '/' + s for s in
                          os.listdir(self.data_path)]
        
        self.on_epoch_end()
        with np.load(self.file_list[0]) as npzfile:
            self.in_dims = []
            self.n_channels = 1
            for i in range(len(self.arrays)):
                im = npzfile[self.arrays[i]]
                self.in_dims.append((self.batch_size,
                                    *np.shape(im),
                                    self.n_channels))

    def __len__(self):
        """Get the number of batches per epoch."""
        return int(np.floor((len(self.file_list)) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.file_list[k] for k in indexes]

        # Generate data
        a = self.__data_generation(list_IDs_temp)
        return a

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self.indexes = np.arange(len(self.file_list))
        np.random.shuffle(self.indexes)
    
    #@threadsafe_generator
    def __data_generation(self, temp_list):
        """Generate data containing batch_size samples."""
        # X : (n_samples, *dim, n_channels)
        # Initialization
        arrays = []

        for i in range(len(self.arrays)):
            arrays.append(np.empty(self.in_dims[i]).astype(np.single))

        for i, ID in enumerate(temp_list):
            with np.load(ID) as npzfile:
                for idx in range(len(self.arrays)):
                    x = npzfile[self.arrays[idx]] \
                        .astype(np.single)
                    x = np.expand_dims(x, axis=2)
                    arrays[idx][i, ] = x

        return arrays


#%% Build model
def build_model():
    
    num_filt = 64                       #HYPERPARAMETER
    in_shape=(128, 128, 1)
    
    inp1 = Input(in_shape)
    inp2 = Input(in_shape)
    inp3 = Input(in_shape)
    inp4 = Input(in_shape)
    
    
    inputs = concatenate([inp1, inp2, inp3, inp4],axis=3) 
    
    
    x = Conv2D(num_filt, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = Conv2D(num_filt, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x) #bn
    x_1 = x

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(num_filt * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(num_filt * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x) #bn
    x_2 = x
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(num_filt * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(num_filt * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x) #bn
    x_3 = x
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(num_filt * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x) #8
    x = Conv2D(num_filt * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)#8
    x = BatchNormalization()(x) #bn
    x_4 = x
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(num_filt * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(num_filt * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x) #bn
    
    x = Conv2D(num_filt * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(x))# 8
    x = concatenate([x_4,x], axis=3)
    x = Conv2D(num_filt * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)#8
    x = Conv2D(num_filt * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)#8
    x = BatchNormalization()(x) #bn

    x = Conv2D(num_filt * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(x))
    x = concatenate([x_3,x], axis=3)
    x = Conv2D(num_filt * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(num_filt * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x) #bn

    x = Conv2D(num_filt * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(x))
    x = concatenate([x_2,x], axis=3)
    x = Conv2D(num_filt * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(num_filt * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x) #bn
    x = Dropout(0.4)(x)                #HYPERPARAMETER

    x = Conv2D(num_filt, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(x))
    x = concatenate([x_1, x], axis=3)
    x = Conv2D(num_filt, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(num_filt, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(1, 1, activation="sigmoid")(x)

    model = Model(inputs=[inp1, inp2, inp3, inp4], outputs=x)
    return model


def unet():
    
    num_filt = 16
    
    in_shape=(128, 128, 1)
    
    inp1 = Input(in_shape)
    inp2 = Input(in_shape)
    inp3 = Input(in_shape)
    inp4 = Input(in_shape)
    
    
    inputs = concatenate([inp1, inp2, inp3, inp4],axis=3) # If you use multiple inputs, you can concatenate them like this
    



    conv1 = Conv2D(num_filt, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(num_filt, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(num_filt*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(num_filt*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(num_filt*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(num_filt*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(num_filt*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(num_filt*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(num_filt*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(num_filt*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.2)(conv5)

    # Expansive Path
    # (Upsampling/decoder part)

    up6 = Conv2D(num_filt*8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(num_filt*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(num_filt*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(num_filt*4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(num_filt*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(num_filt*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(num_filt*2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(num_filt*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(num_filt*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(num_filt, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(num_filt, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(num_filt, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inp1, inp2, inp3, inp4], outputs=conv10)



    return model

##########################################################################
def double_conv_block(x, n_filters):

    # Conv2D then ReLU activation
    x = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    # Conv2D then ReLU activation
    x = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = MaxPooling2D(2)(f)
    p = Dropout(0.3)(p) ############################################################

    return f, p
 
def upsample_block(x, conv_features, n_filters):
    # upsample
    x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate 
    x = concatenate([x, conv_features])
    # dropout
    x = Dropout(0.3)(x) ############################################################
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x

def build_unet_model():

    # inputs
    
    in_shape=(128, 128, 1)
    
    inp1 = Input(in_shape)
    inp2 = Input(in_shape)
    inp3 = Input(in_shape)
    inp4 = Input(in_shape)
    
    
    inputs = concatenate([inp1, inp2, inp3, inp4],axis=3) # If you use multiple inputs, you can concatenate them like this
    

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)

    # outputs
    
    output = Conv2D(1, 1, activation='sigmoid')(u9)
    
    unet_model = Model(inputs=[inp1, inp2, inp3, inp4], outputs=output)

    return unet_model


#%% metrics

##############losses:
    
def dice_metric_batch(y_true, y_pred, smooth=0.01):      # commonly used version. Remember: when calculating results we use this on each image

    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_metric_single(y_true, y_pred, smooth=0.01):        # new version, calculates dice on each image of batch and averages

    # y_pred = tf.where(y_pred >= 0.5, 1, 0)                # maybe better not to binarize here
    # y_pred = K.cast(y_pred, 'float32')
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    
    dice = (2. * intersection + smooth) / (K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) + smooth)
    
    # Return the average Dice coefficient over the batch
    mean_dice = tf.reduce_mean(dice)
    return mean_dice

def dice_loss(y_true, y_pred):
    return 1.0 - dice_metric_single(y_true, y_pred)


# Binary crossentropy
def bcross_loss(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return binary_crossentropy(y_true, y_pred)

#combined loss
def combined_loss(y_true, y_pred, scale=0.3):                   ################HYPERPARAMETER#################

    bce = bcross_loss(y_true, y_pred)
    dsc = dice_loss(y_true, y_pred)
    loss = scale * bce + (1-scale) *dsc 
    
    return loss


############performance metrics:
    
def get_metrics(y_true, y_pred):
    
    num = len(y_pred)
    
    score = list()
    
    dice_scores = []  # I want it to be of len = number of samples

    for i in range(num):
        y_pred_slice = y_pred[i] 
        y_true_slice = y_true[i]  
        
        y_pred_slice[y_pred_slice >= 0.5] = 1      #binarize for dice
        y_pred_slice[y_pred_slice < 0.5] = 0
        
        # I am using dice_metric_batch on each slice, so it returns dice of each slice:
            
        # dice_score = K.get_value(dice_metric_batch(y_pred_slice, y_true_slice, smooth=0.01))  #only for data2 where I dont have empty slice
        dice_score = (1 if np.sum(y_true_slice) == 0 and np.sum(y_pred_slice) == 0 else K.get_value(dice_metric_batch(y_pred_slice, y_true_slice, smooth=0.01)))
        
        dice_scores.append(dice_score)
        
      
    score.append(dice_scores)
    

        
    hd95_scores = []

    for i in range(num):
        y_pred_slice = y_pred[i] 
        y_true_slice = y_true[i] 
        
        y_pred_slice[y_pred_slice >= 0.5] = 1
        y_pred_slice[y_pred_slice < 0.5] = 0
        
        try:
            hd95_score = (0 if np.sum(y_true_slice) == 0 and np.sum(y_pred_slice) == 0 else hd95(y_pred_slice, y_true_slice))
        except:
            hd95_score = 1000
        hd95_scores.append(hd95_score)
    
    score.append(hd95_scores)
    
    
    
     
    hd_scores = []

    for i in range(num):
        y_pred_slice = y_pred[i] 
        y_true_slice = y_true[i]
        
        y_pred_slice[y_pred_slice >= 0.5] = 1
        y_pred_slice[y_pred_slice < 0.5] = 0
        
        try:
            hd_score = (0 if np.sum(y_true_slice) == 0 and np.sum(y_pred_slice) == 0 else hd(y_pred_slice, y_true_slice))
        except: 
            hd_score = 1000
        hd_scores.append(hd_score)
        
     
    score.append(hd_scores)   
     
    ravd_scores = []

    for i in range(num):
        y_pred_slice = y_pred[i] 
        y_true_slice = y_true[i]
        
        y_pred_slice[y_pred_slice >= 0.5] = 1
        y_pred_slice[y_pred_slice < 0.5] = 0
        
        try:
            ravd_score = (0 if np.sum(y_true_slice) == 0 and np.sum(y_pred_slice) == 0 else ravd(y_pred_slice, y_true_slice))
        except:
            ravd_score = 1000
        ravd_scores.append(ravd_score)
        
    score.append(ravd_scores) 
        
    asd_scores = []  
      
    for i in range(num):
        y_pred_slice = y_pred[i] 
        y_true_slice = y_true[i]
        
        y_pred_slice[y_pred_slice >= 0.5] = 1
        y_pred_slice[y_pred_slice < 0.5] = 0
        
        try:
            asd_score = (0 if np.sum(y_true_slice) == 0 and np.sum(y_pred_slice) == 0 else asd(y_pred_slice, y_true_slice))
        except:
            asd_score = 1000
        asd_scores.append(asd_score)
        
    score.append(asd_scores) 
        
    return score


#%% model train

def Model_train(n_epochs, model, gen_train, gen_val, model_save_path):
    
    Tr_Loss = []
    Val_Loss = []
    
    Tr_DSC_batch = []    # the commonly used one, dice over batch
    Val_DSC_batch = []
    
    
    Tr_DSC_single= []  
    Val_DSC_single = []   # new one dice over each image and average
    
    best_val_loss = np.Inf  # Initialize the best validation loss to infinity
    
    for epoch in range(n_epochs):
        
        print('running epoch:', epoch)

        training_loss = []
        validating_loss = []
        
        training_DSC_batch = []
        validating_DSC_batch = []
        
        training_DSC_single = []      
        validating_DSC_single = []    
        
        for idx, (t2f, t2w, t1n, t1c, mask) in enumerate(gen_train):  
            Tr = model.train_on_batch([t2f, t2w, t1n, t1c], mask)
            
            training_loss.append(Tr[0])
            training_DSC_batch.append(Tr[1])
            training_DSC_single.append(Tr[2])  
            
            

        Tr_Loss.append(np.mean(training_loss))
        
        Tr_DSC_batch.append(np.mean(training_DSC_batch))
        
        Tr_DSC_single.append(np.mean(training_DSC_single))  
        

        
        for idx, (t2f, t2w, t1n, t1c, mask) in enumerate(gen_val): 
            V = model.test_on_batch([t2f, t2w, t1n, t1c], mask)
            
            validating_loss.append(V[0])
            validating_DSC_batch.append(V[1])
            validating_DSC_single.append(V[2])  

        epoch_val_loss = np.mean(validating_loss)
            
        Val_Loss.append(epoch_val_loss)  
        
        Val_DSC_batch.append(np.mean(validating_DSC_batch))
        
        Val_DSC_single.append(np.mean(validating_DSC_single)) 
        
        if epoch_val_loss < best_val_loss:
            
            print(f"Validation loss improved from {best_val_loss} to {epoch_val_loss}. Model saved.")
            best_val_loss = epoch_val_loss
            model.save(model_save_path)

    return Tr_Loss, Tr_DSC_batch, Tr_DSC_single, Val_Loss, Val_DSC_batch, Val_DSC_single #return all loss and dsc foe all epochs for plots






#%% Data path

gen_dir = 'D:/First Project/Codes/Experiments/Ex1_New/Data_All/data14/'
# gen_dir = os.path.expanduser('~/ML_pitfalls/Ex1/data/')  #for server


array_labels = ['t2f', 't2w', 't1n', 't1c', 'mask']
batch_size = 20


#%% data: test

gen_test_a = DataGenerator(data_path = gen_dir+'Test/Test_a',
                    arrays=array_labels,
                    batch_size=batch_size)

gen_test_b = DataGenerator(data_path = gen_dir+'Test/Test_b',
                    arrays=array_labels,
                    batch_size=batch_size)


#%% data: train&val case1
print('------------------------------------------------')
print('---------------- Case 2 ------------------------')
print('------------------------------------------------')

gen_train = DataGenerator(data_path = gen_dir+'Case2/Train',
                    arrays=array_labels,
                    batch_size=batch_size)

gen_val = DataGenerator(data_path = gen_dir+'Case2/Validation',
                    arrays=array_labels,
                    batch_size=batch_size)

param = {'learning_rate': 0.00005,
         'n_epochs': 100}

#%% Train model

model_end = build_model()
# model = build_unet_model()
# model = unet()

learning_rate = param['learning_rate']

model_save_path = 'D:/First Project/Codes/Experiments/Ex1_New/results/best_model_case2.h5'

## get the commonly use dice(batch) first:


model_end.compile(loss=combined_loss, metrics=[dice_metric_batch, dice_metric_single], optimizer=Adam(lr=learning_rate))

n_epochs = param['n_epochs']

start_time = time.time()

Tr_Loss, Tr_DSC_batch, Tr_DSC_single, Val_Loss, Val_DSC_batch, Val_DSC_single = Model_train(n_epochs, model_end, gen_train, gen_val, model_save_path)

end_time = time.time()

elapsed_time = end_time-start_time

print('The elapsed time is:', elapsed_time)



#%% predictions
Y_Pred_Train = []
Y_True_Train = []

Y_Pred_Val = []
Y_True_Val = []

Y_Pred_Test_a = []
Y_True_Test_a = []

Y_Pred_Test_b = []
Y_True_Test_b = []

model = load_model(model_save_path, custom_objects={
    'combined_loss': combined_loss,
    'dice_metric_batch': dice_metric_batch,
    'dice_metric_single': dice_metric_single
})

for idx, (t2f, t2w, t1n, t1c, mask) in enumerate(gen_train):
    Pred = model.predict_on_batch([t2f, t2w, t1n, t1c])
    # Pred[Pred >= 0.5] = 1
    # Pred[Pred < 0.5] = 0
    Y_Pred_Train.extend(Pred)
    Y_True_Train.extend(mask)
    
    
for idx, (t2f, t2w, t1n, t1c, mask) in enumerate(gen_val):
    Pred_val = model.predict_on_batch([t2f, t2w, t1n, t1c])
    # Pred_val[Pred_val >= 0.5] = 1
    # Pred_val[Pred_val < 0.5] = 0
    Y_Pred_Val.extend(Pred_val)
    Y_True_Val.extend(mask)


########### test

for idx, (t2f, t2w, t1n, t1c, mask) in enumerate(gen_test_a):
    Pred_tsa = model.predict_on_batch([t2f, t2w, t1n, t1c])
    # Pred_tsa[Pred_tsa >= 0.5] = 1
    # Pred_tsa[Pred_tsa < 0.5] = 0
    Y_Pred_Test_a.extend(Pred_tsa)
    Y_True_Test_a.extend(mask)
    
for idx, (t2f, t2w, t1n, t1c, mask) in enumerate(gen_test_b):
    Pred_tsb = model.predict_on_batch([t2f, t2w, t1n, t1c])
    # Pred_tsb[Pred_tsb >= 0.5] = 1
    # Pred_tsb[Pred_tsb < 0.5] = 0
    Y_Pred_Test_b.extend(Pred_tsb)
    Y_True_Test_b.extend(mask)
    
#%% save y-y_hat

y_yhat_tr = pd.DataFrame({
    'Real': Y_True_Train ,
    'Predicted': Y_Pred_Train
})

y_yhat_val = pd.DataFrame({
    'Real': Y_True_Val ,
    'Predicted': Y_Pred_Val
})

y_yhat_a = pd.DataFrame({
    'Real': Y_True_Test_a ,
    'Predicted': Y_Pred_Test_a
})

y_yhat_b = pd.DataFrame({
    'Real': Y_True_Test_b ,
    'Predicted': Y_Pred_Test_b
})


# y_yhat1_filename = os.path.expanduser('~/ML_pitfalls/Ex1/results/y_yhat_case1.xlsx')
y_yhat_filename = 'D:/First Project/Codes/Experiments/Ex1_New/results/y_yhat_case2.xlsx'
with pd.ExcelWriter(y_yhat_filename, engine='xlsxwriter') as writer:
    y_yhat_tr.to_excel(writer, sheet_name='Train', index=False)
    y_yhat_val.to_excel(writer, sheet_name='Val', index=False)
    y_yhat_a.to_excel(writer, sheet_name='Test_a', index=False)
    y_yhat_b.to_excel(writer, sheet_name='Test_b', index=False)

    
  
#%% Calculate metrics:

train_metric = get_metrics(Y_True_Train, Y_Pred_Train)
val_metric = get_metrics(Y_True_Val, Y_Pred_Val)
test_metric_a = get_metrics(Y_True_Test_a, Y_Pred_Test_a)  
test_metric_b = get_metrics(Y_True_Test_b, Y_Pred_Test_b)  


#%% Calculate uncertainties

def calculate_statistics(metric, conf_level=0.95):

    mean = np.mean(metric)
    std = np.std(metric, ddof=1)   # with(n+1)

    sem = std / np.sqrt(len(metric)) #Standard Error of the Mean (SEM)

    degrees_freedom = len(metric) - 1  #confidence interval
    conf_interval = stats.t.interval(conf_level, degrees_freedom, loc=mean, scale=sem)

    return mean, std, sem, conf_interval


def process_dataset(metrics_data, metric_names, value_to_exclude=1000):


    all_statistics = {}

    for metrics, metric_name in zip(metrics_data, metric_names):
        
        # Remove 1000s
        filtered_metrics = [metric for metric in metrics if metric != value_to_exclude and not math.isnan(metric)]
        
        # Check if left empty:
        if not filtered_metrics:
            print(f"Warning: No data left for {metric_name} after filtering out {value_to_exclude}.")
            continue  
        
        # Calculate the statistics for current metric
        mean, std, sem, conf_interval = calculate_statistics(filtered_metrics)
        all_statistics[metric_name] = {
            'mean': mean,
            'std': std,
            'SEM': sem,
            'confidence_interval': conf_interval,
        }

    
    df_dict = {
        'Metric': [],
        'Mean': [],
        'Std': [],
        'SEM': [],
        'Confidence Interval Low': [],
        'Confidence Interval High': []
    }

    for metric, statistics in all_statistics.items():
        conf_int_low, conf_int_high = statistics['confidence_interval']
        
        df_dict['Metric'].append(metric)
        df_dict['Mean'].append(statistics['mean'])
        df_dict['Std'].append(statistics['std'])
        df_dict['SEM'].append(statistics['SEM'])
        df_dict['Confidence Interval Low'].append(conf_int_low)
        df_dict['Confidence Interval High'].append(conf_int_high)


    df = pd.DataFrame(df_dict)

    return df

#%%

metric_names = ['dice', 'hd95', 'hd', 'ravd', 'asd']

train_df = process_dataset(train_metric, metric_names)
val_df = process_dataset(val_metric, metric_names)
test_df_a = process_dataset(test_metric_a, metric_names)
test_df_b = process_dataset(test_metric_b, metric_names)


# excel_filename = os.path.expanduser('~/ML_pitfalls/Ex1/results/case1_results.xlsx')
excel_filename = 'D:/First Project/Codes/Experiments/Ex1_New/results/case2_results.xlsx'
# excel_filename = 'case1_results.xlsx'
with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
    train_df.to_excel(writer, sheet_name='Training', index=False)
    val_df.to_excel(writer, sheet_name='Validation', index=False)
    test_df_a.to_excel(writer, sheet_name='Test_a', index=False)
    test_df_b.to_excel(writer, sheet_name='Test_b', index=False)
  
#%% plot curves

plt.figure(1)   # dsc
epochs = range(len(Tr_DSC_batch))
 
plt.plot(epochs, Tr_DSC_batch, color='r', label='Train_batch', linewidth=2)
plt.plot(epochs, Val_DSC_batch, color='b', label='Validation_batch', linewidth=2)

plt.plot(epochs, Tr_DSC_single, color='orange', label='Train_single', linewidth=2)
plt.plot(epochs, Val_DSC_single, color='green', label='Validation_single', linewidth=2)
  

plt.xlabel("Epochs", fontsize = 18) 
plt.ylabel("DSC score", fontsize = 18)

plt.title('DSC plot for Case 2', fontsize = 18)
  

plt.legend()
plt.grid(visible=True, which='major', axis='y')
    
# Saving the figure in SVG format

# image_name = os.path.expanduser('~/ML_pitfalls/Ex1/results/dsc_curve_case1.svg')
image_name = 'D:/First Project/Codes/Experiments/Ex1_New/results/dsc_curve_case2.svg'
plt.savefig(image_name, format='svg')

plt.show()

df = pd.DataFrame({
    'Epoch': epochs,
    'Train_DSC_batch': Tr_DSC_batch,
    'Validation_DSC_batch': Val_DSC_batch,
    'Train_DSC_single': Tr_DSC_single,
    'Validation_DSC_single': Val_DSC_single
    
})

# filename = os.path.expanduser('~/ML_pitfalls/Ex1/results/DSC_scores_case1.xlsx')
filename = 'D:/First Project/Codes/Experiments/Ex1_New/results/DSC_scores_case2.xlsx'
df.to_excel(filename, index=False)  


############
plt.figure(2) #loss
epochs = range(len(Tr_Loss))
 
plt.plot(epochs, Tr_Loss, color='r', label='Train', linewidth=2)
plt.plot(epochs, Val_Loss, color='b', label='Validation', linewidth=2)
  

plt.xlabel("Epochs", fontsize = 18) 
plt.ylabel("Loss", fontsize = 18)

plt.title('Loss plot for Case 2', fontsize = 18)
  

plt.legend()
plt.grid(visible=True, which='major', axis='y')
    
# Saving the figure in SVG format

# image_name = os.path.expanduser('~/ML_pitfalls/Ex1/results/dsc_curve_case1.svg')
image_name = 'D:/First Project/Codes/Experiments/Ex1_New/results/Loss_case2.svg'
plt.savefig(image_name, format='svg')

plt.show()

dfLoss = pd.DataFrame({
    'Epoch': epochs,
    'Train_Loss': Tr_Loss,
    'Validation_Loss': Val_Loss
})

# filename = os.path.expanduser('~/ML_pitfalls/Ex1/results/DSC_scores_case1.xlsx')
filename = 'D:/First Project/Codes/Experiments/Ex1_New/results/Loss_case2.xlsx'
df.to_excel(filename, index=False)  



#%%
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# # Assuming Tr_DSC_batch, Val_DSC_batch, Tr_DSC_single, and Val_DSC_single are defined
# epochs = range(len(Tr_DSC_batch))

# # Create a DataFrame for Seaborn
# df = pd.DataFrame({
#     'Epoch': list(epochs) * 4,
#     'DSC Score': Tr_DSC_batch + Val_DSC_batch + Tr_DSC_single + Val_DSC_single,
#     'Type': ['Train_batch'] * len(epochs) + ['Validation_batch'] * len(epochs) + 
#             ['Train_single'] * len(epochs) + ['Validation_single'] * len(epochs)
# })

# # Plotting with Seaborn
# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))

# sns.lineplot(x='Epoch', y='DSC Score', hue='Type', style='Type', data=df, palette=['r', 'b', 'orange', 'green'], linewidth=2)

# plt.xlabel("Epochs", fontsize=18)
# plt.ylabel("DSC Score", fontsize=18)
# plt.title('DSC Plot for Case 2', fontsize=18)
# plt.legend(title='Type')

# # Saving the figure
# image_name = 'D:/First Project/Codes/Experiments/Ex1_New/results/dsc_curve_case2.svg'
# plt.savefig(image_name, format='svg')

# plt.show()

# # Save the DataFrame to Excel
# filename = 'D:/First Project/Codes/Experiments/Ex1_New/results/DSC_scores_case2.xlsx'
# df.to_excel(filename, index=False)
#%%
# plt.figure(3) #loss

# # Assuming Tr_Loss and Val_Loss are defined
# epochs = range(len(Tr_Loss))

# # Create the base plot
# # plt.figure(figsize=(10, 6))

# # Set the background color
# plt.gca().set_facecolor('#003366')  # Dark blue background, change color code as needed

# # Plot your data
# plt.plot(epochs, Tr_Loss, color='r', label='Train', linewidth=2)
# plt.plot(epochs, Val_Loss, color='b', label='Validation', linewidth=2)

# # Set labels and title
# plt.xlabel("Epochs", fontsize=18)
# plt.ylabel("Loss", fontsize=18)
# plt.title('Loss Plot for Case 2', fontsize=18)

# # Legend and grid
# plt.legend()
# plt.grid(visible=True, which='major', axis='y', color='white')

# # Adjust tick colors for visibility
# plt.tick_params(colors='white')

# # Save the figure in SVG format
# image_name = 'D:/First Project/Codes/Experiments/Ex1_New/results/Loss_case2.svg'
# plt.savefig(image_name, format='svg', facecolor='#003366')  # Set the facecolor here as well

# plt.show()

# # Creating DataFrame and saving to Excel (your existing code)
# dfLoss = pd.DataFrame({
#     'Epoch': epochs,
#     'Train_Loss': Tr_Loss,
#     'Validation_Loss': Val_Loss
# })

# filename = 'D:/First Project/Codes/Experiments/Ex1_New/results/Loss_case2.xlsx'
# dfLoss.to_excel(filename, index=False)
