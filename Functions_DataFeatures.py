import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy import signal 
#import emd
import os


def load_data(Folder_fault, all_fault_types, fault_diameters, sel_speed, y_curr, signals_2_extract):

     X_all = []
     Y_all = []
     Y_all_names = {}
     y_last = -2
     curr_signals = []

     for fault_type in all_fault_types:
         for fault_diam in fault_diameters:  
             fault_file_name = "Signals_12K_" + fault_type + "_" + str(fault_diam) + "_" + str(sel_speed) + ".csv"
             if os.path.isfile(Folder_fault + fault_file_name):
                 Signals = pd.read_csv(Folder_fault + fault_file_name)
                 curr_signals = []
                 for sig_name in signals_2_extract:
                     curr_signals.append(Signals[sig_name])
                 curr_signals = np.array(curr_signals)#.transpose()
                 y_last = y_curr + 1
                 Y_all.append(y_last)
                 Y_all_names[y_last] = fault_type + "_" + str(fault_diam)

                 
     return curr_signals, y_last, Y_all_names, y_curr      

def rehsape_2_3d(X, frame_x, frame_y):
    return  np.reshape(X, (X.shape[0], X.shape[1], frame_x, frame_y))

def multiclass_conf_matrix(y_true, y_pred):
    
    n_classes = len(np.unique(y_true))
    conf_matrix = np.zeros((n_classes, n_classes))
    
    # ***
    return conf_matrix

def just_winow_time_signal(X, frame_len, frame_overlap_len, N_fft):
 
    frame_begin = 0
    frame_end = frame_len
    all_frames = []
    while frame_end<len(X):          
            curr_frame = X[frame_begin:frame_end]
            all_frames.append(curr_frame)
            frame_begin += frame_overlap_len
            frame_end = frame_begin + frame_len
            
    return all_frames
            

def fft_analysis(N,Fs,X):
    
    T = 1/Fs
    yf = fft(np.array(X))
    xf = fftfreq(N,T)[:N//2]
  #  plt.plot(xf, 2.0/N*np.abs(yf[0:N//2]))
    
    return xf,yf     


def extract_stft_features(X_all, Y_all, Fs, **kwargs):
    
    
    frame_len = kwargs['frame_len']
    frame_move_len = kwargs['frame_move_len']
    win_len = kwargs['win_len']
    overlap_l = kwargs['overlap_l']
    
    
    N_signals = len(X_all)
    N_ft =  X_all[0].shape[0]
    stft_matrix = []
    Y_matrix = []
    #brojac_signala = 0
    for i  in range(0,N_signals):
        X = X_all[i]
        Y = Y_all[i]
        frame_begin = 0
        frame_end = frame_len
        
        while frame_end < X.shape[1]:

            curr_frame = X[:, frame_begin:frame_end]
            stft_matrix_mini = []   
            for jj in range(N_ft):
             
             f, t, curr_frame_stft = signal.stft(curr_frame[jj,:], Fs, nperseg=win_len, noverlap=overlap_l)

             stft_matrix_mini.append(np.abs(curr_frame_stft))
             
             
            stft_matrix.append(np.array(stft_matrix_mini))
            frame_begin += frame_move_len
            frame_end = frame_begin + frame_len
            Y_matrix.append(Y)
            #print(len(Y_matrix))
            #if len(Y_matrix) == 1214:
            #    print("evo")
            
    return np.array(stft_matrix), np.array(Y_matrix), f, t


def extract_fft_features(X_all, Y_all, Fs, **kwargs):
    
    
    N_fft = kwargs['N_fft']
    frame_len = kwargs['frame_len']
    frame_move_len = kwargs['frame_move_len']
    
    N_signals = len(X_all)
    N_ft =  X_all[0].shape[0]
    fft_matrix = []
    Y_matrix = []
    #brojac_signala = 0
    for i  in range(0,N_signals):
        X = X_all[i]
        Y = Y_all[i]
        frame_begin = 0
        frame_end = frame_len
        
        while frame_end < X.shape[1]:

            curr_frame = X[:, frame_begin:frame_end]
            fft_matrix_mini = []   
            for jj in range(N_ft): 
             
             xf, curr_frame_fft = fft_analysis(N_fft,Fs,curr_frame[jj,:])   
             fft_matrix_mini.append(2.0/N_fft*np.abs(curr_frame_fft[0:N_fft//2]))
             
             
            fft_matrix.append(np.array(fft_matrix_mini))
            frame_begin += frame_move_len
            frame_end = frame_begin + frame_len
            Y_matrix.append(Y)
            #print(len(Y_matrix))
            #if len(Y_matrix) == 1214:
            #    print("evo")
            
    return np.array(fft_matrix), np.array(Y_matrix), xf


def extract_EnvSpec_features(X_all, Y_all, N, Fs, frame_len, frame_move_len):
    
    N_signals = len(X_all)
    fft_matrix = []
    Y_matrix = []
    N = frame_len
    xf = None
    
    for i  in range(0,N_signals):
        if len(X_all) == 1:
            X = X_all[0]
            Y = Y_all[0]
        else:     
            X = X_all[i][0]
            Y = Y_all[i][0]
        
        frame_begin = 0
        frame_end = frame_len
        while frame_end<len(X):

            curr_frame = X[frame_begin:frame_end]
            
            
            analytic_signal = signal.hilbert(curr_frame[:,0])# currently takes only one signal
            amplitude_envelope = np.abs(analytic_signal)
            xf, curr_frame_fft = fft_analysis(N, Fs, amplitude_envelope)           
            
            curr_frame_fft[0] = 0
            fft_matrix.append(2.0/N*np.abs(curr_frame_fft[0:N//2]))
            
            
            frame_begin += frame_move_len
            frame_end = frame_begin + frame_len
            Y_matrix.append(Y)
            
    return np.array(fft_matrix), np.array(Y_matrix), xf


def extract_EnvSpec_featuresCNN(X_all, Y_all, N, Fs, frame_len, frame_move_len):
    
    N_signals = len(X_all)
    fft_matrix = []
    Y_matrix = []
    for i  in range(0,N_signals):
        X = X_all[i][0]
        Y = Y_all[i][0]
        frame_begin = 0
        frame_end = frame_len
        while frame_end<len(X):

            curr_frame = X[frame_begin:frame_end]
            
            
            analytic_signal = signal.hilbert(curr_frame[:,0])# currently takes only one signal
            amplitude_envelope = np.abs(analytic_signal)
            xf, curr_frame_fft = fft_analysis(N, Fs, amplitude_envelope)           
            
            fft_matrix.append(2.0/N*np.abs(curr_frame_fft[0:N//2]))
            
            
            frame_begin += frame_move_len
            frame_end = frame_begin + frame_len
            Y_matrix.append(Y)
           
    fft_matrix = np.array(fft_matrix)        
    fft_matrix = np.reshape(fft_matrix, (len(fft_matrix),8,8,1))  
    return np.array(fft_matrix), np.array(Y_matrix)



def extractTimeFeatures(X_all, Y_all, **kwargs):

    
    frame_len = kwargs['frame_len']
    frame_move_len = kwargs['frame_move_len']
    
    N_signals = len(X_all)
    X_time = []
    Y_matrix = []
    frame_begin = 0
    frame_end = frame_len
    N_ft = X_all[0].shape[1]

    for i  in range(0,N_signals):
        X = X_all[i]
        Y = Y_all[i]
        frame_begin = 0
        frame_end = frame_len
        while frame_end < X.shape[1]:
            
            X_time.append(X[:, frame_begin:frame_end])
            frame_begin += frame_move_len
            frame_end = frame_begin + frame_len
            Y_matrix.append(Y)

    
    return np.array(X_time), np.array(Y_matrix)        


def main_extract_features(X_all, Y_all, chosen_features, Fs, **kwargs):

  kwargs_outputs = {}

  if chosen_features == 'FFT':
        X_feat_all, Y_all, x_f = extract_fft_features(X_all, Y_all, Fs, **kwargs) 
        kwargs_outputs.update({'f':x_f})
  elif chosen_features == 'STFT':
        X_feat_all, Y_all, f, t = extract_stft_features(X_all, Y_all, Fs, **kwargs)
        kwargs_outputs.update({'f':f})
        kwargs_outputs.update({'t':t})
  elif chosen_features == 'TimeFrames': 
        X_feat_all, Y_all =  extractTimeFeatures(X_all, Y_all, **kwargs)  
        X_feat_all = rehsape_2_3d(X_feat_all, kwargs['time_frame_x'], kwargs['time_frame_y'])

  else:
      print("undefined feature type !!!")
      return -1,-1,-1
      
  return X_feat_all, Y_all, kwargs_outputs


