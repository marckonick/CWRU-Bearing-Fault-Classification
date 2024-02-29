import numpy as np
import Functions_DataFeatures as ff
import Function_Models as modata
import time 
import torch
import torch.optim as optim
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import yaml 


def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)

def main():   
    
    config = load_config("config.yaml")  
    df_config = config['data_and_features']
    mtrain_config = config['model_optimization']

    data_folder = df_config['data_folder']
    save_folder = df_config['save_folder']
    sel_speeds = df_config['sel_speed']
    to_save_res = df_config['to_save_res']
    

    all_fault_types = df_config['all_fault_types'] #args.fault_diameterses 
    all_fault_types = [[xx] for xx in all_fault_types]


    fault_diameterses = df_config['fault_diameterses'] #args.fault_diameterses 
    fault_diameterses = [[xx] for xx in fault_diameterses]

    signals_2_extract = df_config['signals_2_extract']
    
    t_start = time.time()
    y_curr=-1
    X_all, Y_all =[],[]
    Y_all_names = {}


    for chosen_class in all_fault_types:      
       for fault_diameters in fault_diameterses:
         for sel_speed in sel_speeds:  
           X_tmp, Y_tmp, Y_name_tmp, y_curr  = ff.load_data(data_folder , chosen_class, fault_diameters, sel_speed, y_curr, signals_2_extract)
           if Y_tmp != -2: # -2 == file_not_found
              X_all.append(X_tmp)
              Y_all.append(Y_tmp)
         if Y_tmp != -2:      
             Y_all_names.update(Y_name_tmp)
             y_curr+=1
     

    kwargs_arguments = {'frame_len':df_config["frame_len"], 'frame_move_len':df_config["frame_move_len"], 'win_len':df_config["win_len_stft"], 'overlap_l':df_config["overlap_l_stft"], 
                    'N_fft':df_config["frame_len"], 'time_frame_x':df_config["time_frame_x"], 'time_frame_y':df_config["time_frame_y"] }

     
    chosen_features = df_config["selected_feature"] #'FFT' # STFT, TimeFrames
    Fs = df_config['Fs']
  
    
    X_all, Y_all, kwargs_outputs = ff.main_extract_features(X_all, Y_all, chosen_features, Fs, **kwargs_arguments)

    t_end = time.time()

    print(f"Elapsed time for feature extracting {chosen_features} features {t_end-t_start} seconds")
    n_classes = len(Y_all_names) 



    ############ Model and optimization ###########################################
    test_size = mtrain_config['test_size']
    batch_size = mtrain_config['batch_size']


    if chosen_features == "FFT": 
        X_all = np.reshape(X_all, (X_all.shape[0], X_all.shape[1]*X_all.shape[2]))
    elif chosen_features == "STFT" or chosen_features == "TimeFrames": 
        in_channels = X_all.shape[1]
     
     
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=test_size, random_state=42)


    X_train = modata.labeled_dataset(X_train, Y_train)
    X_test = modata.labeled_dataset(X_test, Y_test)


    X_train = data.DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    X_test = data.DataLoader(X_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    ################## MODEL AND TRAINING ###################
    device = mtrain_config['device']
    

    
    
    if chosen_features == "FFT": 
        model = modata.DNN_MEL(X_all.shape[1], n_classes=n_classes, n_per_layer=[128,32])
    elif chosen_features == "STFT":  
        model = modata.VGG_1D(n_classes, in_channels=in_channels, n_chans1=[8,16,16, 16], k_size = [3,3,3,3], padding_t='same', fc_size = 384) # 384
    elif chosen_features == "TimeFrames":
        model = modata.VGG_1D(n_classes, in_channels=in_channels, n_chans1=[8,16,16, 16], k_size = [3,3,3,3], padding_t='same', fc_size = 576) # 384


    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()
    n_epochs = mtrain_config['n_epochs']
    model.number_of_params()  # prints number of params

    model.train()
    for epoch in range(1,n_epochs+1):
        loss_train = 0.0
        for x, y in X_train:
               x = x.to(device=device)
               y = y.to(device=device)

               outputs = model(x.float())
               loss = loss_fn(outputs, y.long())

               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               loss_train  += loss.item()
        print(" Epoch ", epoch, "/",n_epochs, " loss = ", float(loss_train/len(X_train)))



########### Testing #################

    Y_names_list = [yy for yy in Y_all_names.values()]
    #np.save('Y_names_list.npy', Y_names_list)
    cm, cm_norm = modata.test_model(X_test, model, n_classes, Y_names_list, device)
    
    ts = str(int(time.time()))
    
    if to_save_res:
     sv = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, fmt = 'd', xticklabels=Y_names_list, yticklabels=Y_names_list)
     figure = sv.get_figure()    
     figure.savefig(save_folder + 'cm_' + chosen_features + '_' + '_'.join(map(str, sel_speeds)) +  '_' + ts  + '.png', dpi=400) 
     sv.cla()
    
     sv = sns.heatmap(cm_norm, annot=True, cmap='Blues', cbar=False, xticklabels=Y_names_list, yticklabels=Y_names_list)
     figure = sv.get_figure()    
     figure.savefig(save_folder + 'norm_cm_' + chosen_features + '_' + '_'.join(map(str, sel_speeds)) + '_' + ts + '.png', dpi=400) 

 
if __name__ == '__main__':
    main()

#cm_stft_speeds_1750_1772_1797
