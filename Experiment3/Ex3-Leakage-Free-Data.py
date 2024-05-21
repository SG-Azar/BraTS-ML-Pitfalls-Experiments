# -*- coding: utf-8 -*-
"""
Created on Thu Dec 2 11:52:55 2023


Experiment3:
    
    
    This code extracts the Leakage-Free-Data


@author: sgazar
"""


#%% Cell 1

import os
import pandas as pd
import numpy as np
# from sklearn.preprocessing import scale
from nibabel import load as load_nii
import cv2
import random 

#%% seed
seed = 4    

random.seed(seed)
np.random.seed(seed)

#%% Cell 2: Read mapping file

mappingfile_path = 'D:\First Project\Data\BraTs2023\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'

mappingfile = pd.read_excel(os.path.join( mappingfile_path, 'BraTS2023_2017_GLI_Mapping.xlsx'))

cite_numbers = mappingfile['Site No (represents the originating institution)'].value_counts()

#%% Data path for saving

dataset = 'D:/First Project/Data/BraTs2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/'
save_dir = 'D:/First Project/Codes/Experiments/Ex3/data/case3/NoLeakage/'

#%% params

tr_num = 60
val_num = 20
test_num = 20

active_sites = 7   # the first seven sites with the most number of patients

total = tr_num + val_num + test_num   

site_indexes = list(cite_numbers.index[0:active_sites])  # the first seven sites

#%% Ex test
# Test b
test_site_b = mappingfile[mappingfile['Site No (represents the originating institution)'] == site_indexes[0]]  # site 18

test_names_b = list(test_site_b['BraTS2023'])

random_Test_b = random.sample(test_names_b ,test_num)



####### for Test b:
Ex_save_dir = 'D:/First Project/Codes/Experiments/Ex3/data/case3/Ex_test/'
    
img_size = 128

patients = random_Test_b

pat_idx = 0

for patient in patients:
    
    
    save_path = Ex_save_dir

    pat_idx = pat_idx + 1
    
    for dirName, subdirList, fileList in os.walk(os.path.join(dataset,patient)):
        t2f = load_nii(os.path.join(dirName, patient + '-t2f.nii.gz')).get_fdata()
        t2w = load_nii(os.path.join(dirName, patient + '-t2w.nii.gz')).get_fdata()
        t1n = load_nii(os.path.join(dirName, patient + '-t1n.nii.gz')).get_fdata()
        t1c = load_nii(os.path.join(dirName, patient + '-t1c.nii.gz')).get_fdata()
        labels = load_nii(os.path.join(dirName, patient + '-seg.nii.gz')).get_data()
        labels = labels.astype(np.uint8)
        
        slice_size = np.shape(labels)[-1] - 55
        for slc in range(0, slice_size, 2):     #in range(slice_size):
            slc_idx = slc + 27
            t2f_slice = np.array(cv2.resize(t2f[:, :, slc_idx] / np.max(t2f[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t2w_slice = np.array(cv2.resize(t2w[:, :, slc_idx] / np.max(t2w[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t1n_slice = np.array(cv2.resize(t1n[:, :, slc_idx] / np.max(t1n[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            t1c_slice = np.array(cv2.resize(t1c[:, :, slc_idx] / np.max(t1c[:, :, slc_idx]), dsize=(img_size, img_size)), dtype=np.single)
            labels_slice = np.array(cv2.resize(labels[:, :, slc_idx], dsize=(img_size, img_size), interpolation=cv2.INTER_NEAREST) > 0, dtype=np.bool)
            

            np.savez(save_path + patient + '_' + str(slc),
                     t2f=t2f_slice,
                     t2w=t2w_slice,
                     t1n=t1n_slice,
                     t1c=t1c_slice,
                     mask=labels_slice)



#%% ################ train and Validation sets ################# 



site_num = 16   


case2_site1 = mappingfile[mappingfile['Site No (represents the originating institution)'] == site_indexes[1]]
case2_site2 = mappingfile[mappingfile['Site No (represents the originating institution)'] == site_indexes[2]]
case2_site3 = mappingfile[mappingfile['Site No (represents the originating institution)'] == site_indexes[3]]
case2_site4 = mappingfile[mappingfile['Site No (represents the originating institution)'] == site_indexes[4]]
case2_site5 = mappingfile[mappingfile['Site No (represents the originating institution)'] == site_indexes[5]]
case2_site6 = mappingfile[mappingfile['Site No (represents the originating institution)'] == site_indexes[6]]


case2_names_all1 = list(case2_site1['BraTS2023'])
case2_names_all2 = list(case2_site2['BraTS2023'])
case2_names_all3 = list(case2_site3['BraTS2023'])
case2_names_all4 = list(case2_site4['BraTS2023'])
case2_names_all5 = list(case2_site5['BraTS2023'])
case2_names_all6 = list(case2_site6['BraTS2023'])



case2_random1 = random.sample(case2_names_all1, site_num+1)
case2_random2 = random.sample(case2_names_all2, site_num+1)
case2_random3 = random.sample(case2_names_all3, site_num+1)
case2_random4 = random.sample(case2_names_all4, site_num+1)
case2_random5 = random.sample(case2_names_all5, site_num)
case2_random6 = random.sample(case2_names_all6, site_num)

case2_names = case2_random1 + case2_random2 + case2_random3 + case2_random4 + case2_random5 + case2_random6



##########################################################


patients = random.sample(case2_names, len(case2_names))

pat_idx = 0

for patient in patients:
    
    
    if pat_idx < tr_num:
        save_path = save_dir + 'Train/'
    elif pat_idx < tr_num + val_num:
        save_path = save_dir + 'Validation/'
    else:
        save_path = save_dir + 'Test/'
        
    pat_idx = pat_idx + 1
    
    
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
            
            np.savez(save_path + patient + '_' + str(slc),
                     t2f=t2f_slice,
                     t2w=t2w_slice,
                     t1n=t1n_slice,
                     t1c=t1c_slice,
                     mask=labels_slice)
        
   
        
        
        
        