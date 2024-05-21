
"""
Created on Mon Oct  9 10:42:37 2023

This code extracts data for Case3 in Experiment 2:


Without adding any confounder

patients are selecting randomly in terms of site info


@author: sgazar
"""

#%% Cell 1

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from nibabel import load as load_nii
import cv2
import random 

#%% seed
seed = 32

random.seed(seed)
np.random.seed(seed)


#%% Cell 2: Read mapping file

mappingfile_path = 'D:\First Project\Data\BraTs2023\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'

labelfile_path = 'C:\BraTs Data\BraTs2021'

mappingfile = pd.read_excel(os.path.join( mappingfile_path, 'BraTS2023_2017_GLI_Mapping.xlsx'))

labelfile = pd.read_csv(os.path.join( labelfile_path, 'train_labels.csv'))


#%% 
mappingfile['numeric_BraTS2021'] = mappingfile['BraTS2021'].str.extract('(\d+)$').astype(int)

merged = pd.merge(labelfile, mappingfile, left_on='BraTS21ID', right_on='numeric_BraTS2021')

final = merged[['BraTS2023', 'MGMT_value', 'Site No (represents the originating institution)']]


#%% sample

num = 45   
tr = 25    
val = 10  
ts = 10  

seed = 32

# Select "num" random patients with MGMT_value of 0
zero_sample = final[(final['MGMT_value'] == 0)].sample(num, random_state=seed)


# Select "num" random patients with MGMT_value of 1 
one_sample = final[(final['MGMT_value'] == 1)].sample(num, random_state=seed)



#For zero_sample
zero_train = zero_sample.sample(tr, random_state=seed)
zero_sample = zero_sample.drop(zero_train.index)
zero_val = zero_sample.sample(val, random_state=seed)
zero_test = zero_sample.drop(zero_val.index)


# For one_sample
one_train = one_sample.sample(tr, random_state=seed)
one_sample = one_sample.drop(one_train.index)
one_val = one_sample.sample(val, random_state=seed)
one_test = one_sample.drop(one_val.index)

train_df = pd.concat([zero_train, one_train])
val_df = pd.concat([zero_val, one_val])
test_df = pd.concat([zero_test, one_test])

#%% Shuffle
train_df = train_df.sample(frac=1, random_state=seed)
val_df = val_df.sample(frac=1, random_state=seed)
test_df = test_df.sample(frac=1, random_state=seed)



#%% Create an external test set

# Dropping rows that have already been selected
remaining_df = final.drop(train_df.index).drop(val_df.index).drop(test_df.index)

# Select one row with MGMT_value of 0 and Site not equal to 1
external_test_zero = remaining_df[(remaining_df['MGMT_value'] == 0)].sample(ts, random_state=seed)

# Select one row with MGMT_value of 1 and Site not equal to 1
external_test_one = remaining_df[(remaining_df['MGMT_value'] == 1)].sample(ts, random_state=seed)

# Combine the two samples
external_test_df = pd.concat([external_test_zero, external_test_one])
external_test_df = external_test_df.sample(frac=1, random_state=seed)
#%% Data path for saving

dataset = 'D:/First Project/Data/BraTs2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/'
save_dir = 'D:/First Project/Codes/Experiments/Ex2/data/data1-clean-small/'


#%% add noise function


def add_gaussian_noise(image, mean=0, std_dev=0.1):

    noise = np.random.normal(loc=mean, scale=std_dev, size=image.shape)
    noisy_image = np.clip(image + noise, 0, 1)  # Ensure pixel values are valid
    return noisy_image

#%% preprocess and save data:

########## Train
# original images are 240x240x155
img_size = 128

patients = train_df['BraTS2023']

pat_idx = 0

for patient in patients:
    
    
    
    save_path = save_dir + 'Train/'

    pat_idx = pat_idx + 1
    
    
    mgmt_value = train_df[train_df['BraTS2023'] == patient]['MGMT_value'].values[0]
    
    for dirName, subdirList, fileList in os.walk(os.path.join(dataset,patient)):
        t2f = load_nii(os.path.join(dirName, patient + '-t2f.nii.gz')).get_fdata()
        t2w = load_nii(os.path.join(dirName, patient + '-t2w.nii.gz')).get_fdata()
        t1n = load_nii(os.path.join(dirName, patient + '-t1n.nii.gz')).get_fdata()
        t1c = load_nii(os.path.join(dirName, patient + '-t1c.nii.gz')).get_fdata()
        labels = load_nii(os.path.join(dirName, patient + '-seg.nii.gz')).get_data()
        labels = labels.astype(np.uint8)

        
        slice_size = np.shape(labels)[-1] - 55  
        
        for slc in range(0, slice_size, 2):
            slc_idx = slc + 27
            t2f_slice = np.array(cv2.resize(t2f[:, :, slc_idx] / np.max(t2f[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t2w_slice = np.array(cv2.resize(t2w[:, :, slc_idx] / np.max(t2w[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t1n_slice = np.array(cv2.resize(t1n[:, :, slc_idx] / np.max(t1n[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t1c_slice = np.array(cv2.resize(t1c[:, :, slc_idx] / np.max(t1c[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            labels_slice = np.array(cv2.resize(labels[:, :, slc_idx], dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST) > 0, dtype=np.bool)
            
            if np.sum(labels_slice) > 0:

            
                np.savez(save_path + patient + '_' + str(slc),
                         t2f=t2f_slice,
                         t2w=t2w_slice,
                         t1n=t1n_slice,
                         t1c=t1c_slice,
                         mask=labels_slice,
                         mgmt=mgmt_value)



#%%

########## Validation
# original images are 240x240x155
img_size = 128

patients = val_df['BraTS2023']

pat_idx = 0

for patient in patients:
    
    
    
    save_path = save_dir + 'Validation/'

    pat_idx = pat_idx + 1
    
    
    mgmt_value = val_df[val_df['BraTS2023'] == patient]['MGMT_value'].values[0]
    
    for dirName, subdirList, fileList in os.walk(os.path.join(dataset,patient)):
        t2f = load_nii(os.path.join(dirName, patient + '-t2f.nii.gz')).get_fdata()
        t2w = load_nii(os.path.join(dirName, patient + '-t2w.nii.gz')).get_fdata()
        t1n = load_nii(os.path.join(dirName, patient + '-t1n.nii.gz')).get_fdata()
        t1c = load_nii(os.path.join(dirName, patient + '-t1c.nii.gz')).get_fdata()
        labels = load_nii(os.path.join(dirName, patient + '-seg.nii.gz')).get_data()
        labels = labels.astype(np.uint8)

        
        slice_size = np.shape(labels)[-1] - 55 
        
        for slc in range(0, slice_size, 2):
            slc_idx = slc + 27
            t2f_slice = np.array(cv2.resize(t2f[:, :, slc_idx] / np.max(t2f[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t2w_slice = np.array(cv2.resize(t2w[:, :, slc_idx] / np.max(t2w[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t1n_slice = np.array(cv2.resize(t1n[:, :, slc_idx] / np.max(t1n[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t1c_slice = np.array(cv2.resize(t1c[:, :, slc_idx] / np.max(t1c[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            labels_slice = np.array(cv2.resize(labels[:, :, slc_idx], dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST) > 0, dtype=np.bool)
            
            if np.sum(labels_slice) > 0:  
            
                np.savez(save_path + patient + '_' + str(slc),
                         t2f=t2f_slice,
                         t2w=t2w_slice,
                         t1n=t1n_slice,
                         t1c=t1c_slice,
                         mask=labels_slice,
                         mgmt=mgmt_value)


#%%

########## Test
# original images are 240x240x155
img_size = 128

patients = test_df['BraTS2023']

pat_idx = 0

for patient in patients:
    
    
    
    save_path = save_dir + 'Test/'

    pat_idx = pat_idx + 1
    
    
    mgmt_value = test_df[test_df['BraTS2023'] == patient]['MGMT_value'].values[0]
    
    for dirName, subdirList, fileList in os.walk(os.path.join(dataset,patient)):
        t2f = load_nii(os.path.join(dirName, patient + '-t2f.nii.gz')).get_fdata()
        t2w = load_nii(os.path.join(dirName, patient + '-t2w.nii.gz')).get_fdata()
        t1n = load_nii(os.path.join(dirName, patient + '-t1n.nii.gz')).get_fdata()
        t1c = load_nii(os.path.join(dirName, patient + '-t1c.nii.gz')).get_fdata()
        labels = load_nii(os.path.join(dirName, patient + '-seg.nii.gz')).get_data()
        labels = labels.astype(np.uint8)

        
        slice_size = np.shape(labels)[-1] - 55 
        
        for slc in range(0, slice_size, 2):
            slc_idx = slc + 27
            t2f_slice = np.array(cv2.resize(t2f[:, :, slc_idx] / np.max(t2f[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t2w_slice = np.array(cv2.resize(t2w[:, :, slc_idx] / np.max(t2w[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t1n_slice = np.array(cv2.resize(t1n[:, :, slc_idx] / np.max(t1n[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t1c_slice = np.array(cv2.resize(t1c[:, :, slc_idx] / np.max(t1c[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            labels_slice = np.array(cv2.resize(labels[:, :, slc_idx], dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST) > 0, dtype=np.bool)
            
            if np.sum(labels_slice) > 0:
                
            
                np.savez(save_path + patient + '_' + str(slc),
                         t2f=t2f_slice,
                         t2w=t2w_slice,
                         t1n=t1n_slice,
                         t1c=t1c_slice,
                         mask=labels_slice,
                         mgmt=mgmt_value)


#%%

########## External Test
# original images are 240x240x155
img_size = 128

patients = external_test_df['BraTS2023']

pat_idx = 0

for patient in patients:
    
    
    
    save_path = save_dir + 'External_Test/'

    pat_idx = pat_idx + 1
    
    
    mgmt_value = external_test_df[external_test_df['BraTS2023'] == patient]['MGMT_value'].values[0]
    
    for dirName, subdirList, fileList in os.walk(os.path.join(dataset,patient)):
        t2f = load_nii(os.path.join(dirName, patient + '-t2f.nii.gz')).get_fdata()
        t2w = load_nii(os.path.join(dirName, patient + '-t2w.nii.gz')).get_fdata()
        t1n = load_nii(os.path.join(dirName, patient + '-t1n.nii.gz')).get_fdata()
        t1c = load_nii(os.path.join(dirName, patient + '-t1c.nii.gz')).get_fdata()
        labels = load_nii(os.path.join(dirName, patient + '-seg.nii.gz')).get_data()
        labels = labels.astype(np.uint8)

        
        slice_size = np.shape(labels)[-1] - 55 

        for slc in range(0, slice_size, 2):
            slc_idx = slc + 27
            t2f_slice = np.array(cv2.resize(t2f[:, :, slc_idx] / np.max(t2f[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t2w_slice = np.array(cv2.resize(t2w[:, :, slc_idx] / np.max(t2w[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t1n_slice = np.array(cv2.resize(t1n[:, :, slc_idx] / np.max(t1n[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t1c_slice = np.array(cv2.resize(t1c[:, :, slc_idx] / np.max(t1c[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            labels_slice = np.array(cv2.resize(labels[:, :, slc_idx], dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST) > 0, dtype=np.bool)
            
            if np.sum(labels_slice) > 0:
                
            
                np.savez(save_path + patient + '_' + str(slc),
                          t2f=t2f_slice,
                          t2w=t2w_slice,
                          t1n=t1n_slice,
                          t1c=t1c_slice,
                          mask=labels_slice,
                          mgmt=mgmt_value)













