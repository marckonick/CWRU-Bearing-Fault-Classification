# CWRU-Bearing-Fault-Classification
Classification of the faults from the the Electrical Engineering Laboratory of Case
Western Reserve University (CWRU) bearing fault dataset. 


## Description 
CWRU Dataset: Vibration data of rolling bearing
More details @ https://engineering.case.edu/bearingdatacenter/welcome
Dataset available @ https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data

## Files Description
- main_classification.py       - script for load data, feature extraction and model training 
- Functions_DataFeatures.py    - functions for data load and feature extraction 
- Function_Models.py           - model definitions

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

