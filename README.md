# BraTS-ML-Pitfalls-Experiments

This repository hosts the source code and data processing scripts used in the paper "From Promise to Practice: A Study of Common Pitfalls Behind the Generalization Gap in Machine Learning". The experiments are designed to illustrate the impacts of three common pitfalls in machine learning processes, specifically focusing on dataset shift, confounders, and information leakage using the Brain Tumor Segmentation (BraTS) dataset.

## Experiments Overview
### Experiment 1: Dataset Shift
This experiment investigates the effects of dataset shift in medical imaging, using MRI scans from different medical centers to study how changes in data distribution affect model performance. We constructed different scenarios to simulate dataset shift and its impact on model generalization.

### Experiment 2: Confounders
This experiment explores how confounders, variables that influence both predictors and outcomes, can bias model predictions. By manipulating MRI data with artificial noise and site-specific biases, we demonstrate how confounders distort accuracy and reliability, underscoring the need for careful confounder identification and adjustment in machine learning models.

### Experiment 3: Information Leakage
This experiment examines how improper data handling practices, such as incorrect data slicing and preprocessing, can lead to information leakage. We demonstrate how such pitfalls can falsely enhance model performance by leaking information from the test set into the training process.

## Dataset
The experiments utilize the BraTS 2023 dataset, which can be accessed in the following link:

https://www.synapse.org/#!Synapse:syn51156910/wiki/621282
