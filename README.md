# CWRU-Bearing-Fault-Classification
Classification of the faults from the the Electrical Engineering Laboratory of Case
Western Reserve University (CWRU) bearing fault dataset. 


## Description 
CWRU Dataset: Vibration data of rolling bearing
More details @ https://engineering.case.edu/bearingdatacenter/welcome
Dataset available @ https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data




## Files Description
- ReadAndConvertFiles.m        - MATLAB file to convert original data files to .csv 
- main_classification.py       - script for load data, feature extraction and model training 
- Functions_DataFeatures.py    - functions for data load and feature extraction 
- Function_Models.py           - model definitions
- Config.yaml                  - configuration file for selecting feautres and model 

## Features
Input signals are divided into overlaping segments for feature extraction.

Feature types:
- FFT - Fast Fourier Transform
- STFT - Short-Time Fourier Transform
- TimeFrames - Time-domain signals

## Models
Model types:
 - Feed-forward neural networks with two hidden layers
 - Convolutional neural netowrks based on VGG architecture


 ## Results
Classification experiment:
    - Total 10 classes 1 normal state and 9 faulty states
    - Each fault corresponds to one fault location and one fault size 
      for all loads (motor speeds)


![stft_results](https://github.com/marckonick/CWRU-Bearing-Fault-Classification/master/Results/cm_STFT_1730_1750_1772_1797_1708227224.png)





