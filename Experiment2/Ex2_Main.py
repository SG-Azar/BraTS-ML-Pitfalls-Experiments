# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:37:26 2023

##########################################################################################

Experiment2:classification for all the cases
    
##########################################################################################
    


@author: sgazar
"""

#%% Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.utils import resample

import tensorflow.keras as keras
from tensorflow.keras.models import Model
from keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Dropout,\
                                    Add, AveragePooling2D, Flatten, Dense, UpSampling2D,\
                                    MaxPooling2D, GlobalAveragePooling2D, Activation, concatenate

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
            for i, array_label in enumerate(self.arrays):
                if array_label != 'mgmt':  # images
                    im = npzfile[array_label]
                    self.in_dims.append((self.batch_size, *np.shape(im), self.n_channels))
                else:  # 'mgmt' scalar
                    self.in_dims.append((self.batch_size, 1))  # shape for scalar

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
        arrays = []
    
        for i in range(len(self.arrays) - 1):  # excluding 'mgmt'
            arrays.append(np.empty(self.in_dims[i]).astype(np.single))
    
        mgmt_values = np.empty((self.batch_size, 1))  # Adding this line to handle mgmt values
    
        for i, ID in enumerate(temp_list):
            with np.load(ID) as npzfile:
                for idx in range(len(self.arrays) - 1):  # excluding 'mgmt'
                    x = npzfile[self.arrays[idx]].astype(np.single)
                    x = np.expand_dims(x, axis=2)
                    arrays[idx][i, ] = x
    
                # Handling the mgmt value separately
                mgmt_values[i] = npzfile['mgmt'].astype(np.single)
    
        arrays.append(mgmt_values)  # appending mgmt values to batched data
        return arrays

#%%Hyperparameters:
param = {'num_filt': 8,
         'dropout_rate':  0.5,
         'learning_rate': 0.0003,
         'n_epochs': 100} 

#%% Build model
def classification_model(param):
    
    num_filt = param['num_filt']
    dropout_rate = param['dropout_rate']
    in_shape=(128, 128, 1)
    
    inp1 = Input(in_shape)
    inp2 = Input(in_shape)
    inp3 = Input(in_shape)
    inp4 = Input(in_shape)
    
    
    inputs = concatenate([inp1, inp2, inp3, inp4],axis=3) 
    
    
    x = Conv2D(num_filt, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(num_filt, 3, activation='relu', padding='same')(x)
    x_1 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    
    x = Conv2D(num_filt * 2, 3, activation='relu', padding='same')(x)
    x = Conv2D(num_filt * 2, 3, activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x_2 = x
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(num_filt * 4, 3, activation='relu', padding='same')(x)
    x = Conv2D(num_filt * 4, 3, activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x_3 = x
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(num_filt * 4, 3, activation='relu', padding='same')(x)
    x = Conv2D(num_filt * 4, 3, activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x_4 = x
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(num_filt * 16, 3, activation='relu', padding='same')(x)
    x = Conv2D(num_filt * 16, 3, activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Conv2D(num_filt * 4, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x))
    x = concatenate([x_4,x], axis=3)
    x = Conv2D(num_filt * 4, 3, activation='relu', padding='same')(x)
    x = Conv2D(num_filt * 4, 3, activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filt * 4, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x))
    x = concatenate([x_3,x], axis=3)
    x = Conv2D(num_filt * 4, 3, activation='relu', padding='same')(x)
    x = Conv2D(num_filt * 4, 3, activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filt * 2, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x))
    x = concatenate([x_2,x], axis=3)
    x = Conv2D(num_filt * 2, 3, activation='relu', padding='same')(x)
    x = Conv2D(num_filt * 2, 3, activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filt, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x))
    x = concatenate([x_1, x], axis=3)
    x = Conv2D(num_filt, 3, activation='relu', padding='same')(x)
    x = Conv2D(num_filt, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    
    # Classifier:
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(32, activation='relu')(x) 
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x) 
    
    model = Model(inputs=[inp1, inp2, inp3, inp4], outputs=output)
        
    return model





#%% Data path
# gen_dir = 'D:/First Project/Codes/Experiments/Ex2/data/data1-clean-small/'   ##### clean with no site info  --- CASE3

# gen_dir = 'D:/First Project/Codes/Experiments/Ex2/data/data2-clean-small-siteinfo/'  ##### clean with site info  --- CASE2

gen_dir = 'D:/First Project/Codes/Experiments/Ex2/data/data3-noisy-small-siteinfo-sd0007/'  ##### noisy with site info -- CASE1

#%%

array_labels = ['t2f', 't2w', 't1n', 't1c','mgmt'] 
batch_size = 20

#%% Read data

gen_train = DataGenerator(data_path = gen_dir+'Train',
                    arrays=array_labels,
                    batch_size=batch_size)

gen_val = DataGenerator(data_path = gen_dir+'Validation',
                    arrays=array_labels,
                    batch_size=batch_size)

gen_test = DataGenerator(data_path = gen_dir+'Test',
                    arrays=array_labels,
                    batch_size=batch_size)

gen_ex_test = DataGenerator(data_path = gen_dir+'External_Test',
                    arrays=array_labels,
                    batch_size=batch_size)



#%% Train function
n_epochs = param['n_epochs']

def Model_train(n_epochs, model, model_save_path):
    Tr_Loss = []
    Val_Loss = []
    Tr_acc = []
    Val_acc = []
    
    best_val_loss = np.Inf 
    
    for epoch in range(n_epochs):
        print('Running epoch:', epoch)

        training_loss = []
        validating_loss = []
        training_acc = []
        validating_acc = []
        
        for idx, (t2f, t2w, t1n, t1c, mgmt) in enumerate(gen_train):  
            Tr1 = model.train_on_batch([t2f, t2w, t1n, t1c], mgmt)
            training_loss.append(Tr1[0])
            training_acc.append(Tr1[1])
        
        Tr_Loss.append(np.mean(training_loss))
        
        Tr_acc.append(np.mean(training_acc))
        
        

        
        for idx, (t2f, t2w, t1n, t1c, mgmt) in enumerate(gen_val): 
            V1 = model.test_on_batch([t2f, t2w, t1n, t1c], mgmt)
            validating_loss.append(V1[0])
            validating_acc.append(V1[1])
            
            
        Val_acc.append(np.mean(validating_acc))
    
        epoch_val_loss = np.mean(validating_loss)
        Val_Loss.append(epoch_val_loss)
             
        if epoch_val_loss < best_val_loss:
            
            print(f"Validation loss improved from {best_val_loss} to {epoch_val_loss}. Model saved.")
            best_val_loss = epoch_val_loss
            model.save(model_save_path)
        

    return Tr_acc, Val_acc


#%% Train model
save_path = 'D:/First Project/Codes/Experiments/Ex2/results/New Results/Case1'  

model_save_path = os.path.join(save_path, 'model.h5')

model = classification_model(param)


learning_rate = param['learning_rate']

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=learning_rate))

model.summary()

Tr_acc, Val_acc = Model_train(n_epochs, model, model_save_path)

#%% plot DSC


plt.figure(1)   # dsc
epochs = range(len(Tr_acc))
 
plt.plot(epochs, Tr_acc, color='r', label='Train Accuracy', linewidth=2)
plt.plot(epochs, Val_acc, color='b', label='Validation Accuracy', linewidth=2)


plt.xlabel("Epochs", fontsize = 18) 
plt.ylabel("Accuracy", fontsize = 18)


plt.legend()
plt.grid(visible=True, which='major', axis='y')
    
image_name = os.path.join(save_path, 'acc_curve.eps')

plt.savefig(image_name, format='eps')

plt.show()

df = pd.DataFrame({
    'Epoch': epochs,
    'Train_ACC': Tr_acc,
    'Validation_ACC': Val_acc
    
})

filename = os.path.join(save_path, 'acc_scores.xlsx')
df.to_excel(filename, index=False)  



#%%
    
def evaluate_model(model, generator):
    y_true = []
    y_pred = []


    for idx, (t2f, t2w, t1n, t1c, mgmt) in enumerate(generator):
        predictions = model.predict_on_batch([t2f, t2w, t1n, t1c])  
        y_pred.extend(predictions) 
        y_true.extend(mgmt) 

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_true, np.round(y_pred))
    precision = precision_score(y_true, np.round(y_pred))
    recall = recall_score(y_true, np.round(y_pred))
    f1 = f1_score(y_true, np.round(y_pred))
    auc = roc_auc_score(y_true, y_pred)

    return accuracy, precision, recall, f1, auc    


########################


def bootstrap_evaluate(model, generator, n_iterations=1000):

    accuracy_scores, precision_scores, recall_scores, f1_scores, auc_scores = [], [], [], [], []

    X, y = [], []
    for idx, (t2f, t2w, t1n, t1c, mgmt) in enumerate(generator):
        X.append([t2f, t2w, t1n, t1c])
        y.append(mgmt)

    for i in range(n_iterations):

        indices = resample(np.arange(len(y)), replace=True, n_samples=len(y))
        X_sample = [np.vstack([X[i][j] for i in indices]) for j in range(4)] 
        y_sample = np.vstack([y[i] for i in indices]).flatten()
        

        # Evaluate the model
        predictions = model.predict_on_batch(X_sample)
        accuracy = accuracy_score(y_sample, np.round(predictions))
        precision = precision_score(y_sample, np.round(predictions))
        recall = recall_score(y_sample, np.round(predictions))
        f1 = f1_score(y_sample, np.round(predictions))
        auc = roc_auc_score(y_sample, predictions)

        # Store the results
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)

    # Calculate the uncertainty as the standard deviation
    
    accuracy_mean = np.mean(accuracy_scores)
    precision_mean = np.mean(precision_scores)
    recall_mean = np.mean(recall_scores)
    f1_mean = np.mean(f1_scores)
    auc_mean = np.mean(auc_scores)
    
    accuracy_se = np.std(accuracy_scores) / np.sqrt(n_iterations)
    precision_se = np.std(precision_scores) / np.sqrt(n_iterations)
    recall_se = np.std(recall_scores) / np.sqrt(n_iterations)
    f1_se = np.std(f1_scores) / np.sqrt(n_iterations)
    auc_se = np.std(auc_scores) / np.sqrt(n_iterations)
    
    Result = {
        'accuracy': {'mean': accuracy_mean, 'se': accuracy_se},
        'precision': {'mean': precision_mean, 'se': precision_se},
        'recall': {'mean': recall_mean, 'se': recall_se},
        'f1': {'mean': f1_mean, 'se': f1_se},
        'auc': {'mean': auc_mean, 'se': auc_se}
    }

    return Result

    
#%% Evaluate
    
best_model = load_model(model_save_path)


Test_Result = bootstrap_evaluate(best_model, gen_test)
Ex_Test_Result = bootstrap_evaluate(best_model, gen_ex_test)


#%% save


test_results_df = pd.DataFrame(Test_Result)
ex_test_results_df = pd.DataFrame(Ex_Test_Result)

combined_results_df = pd.concat([test_results_df, ex_test_results_df], keys=['Test', 'Extended Test'])

excel_file_path = os.path.join(save_path, 'results.xlsx')
combined_results_df.to_excel(excel_file_path)




































