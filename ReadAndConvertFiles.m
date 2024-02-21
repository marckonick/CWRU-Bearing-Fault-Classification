clear all
close all
clc


FaultClassFolder = 'Fault12K\'; % folder with original .mat files 
all_fault_types = {'Normal','IRF', 'BF', 'OR_C6', 'OR_O3', 'OR_OP12'};
Fault_SampRate = '12K'; 
sel_speed = 1730; % 1730, 1750, 1772, 1790
sel_diam = '0007\'; % 0014 0021 0028

for i_f = 1:length(all_fault_types)
    
    sel_fault = all_fault_types{i_f};  
    T = readtable('SignalFileNames_12K.xlsx');
%% find file name
    [sel_row,a] = find(T.ApproxMotorSpeed==sel_speed & T.FaultDiameter == str2double(sel_diam(1:end-1))/1000); % selecting rows str2double(sel_diam(1:end-1))
    % selecting cells based on row, columns (fault type)
    if strcmp(sel_fault,'IRF')==1
        sel_file_names = T.File_IR(sel_row); 
        FaultTypeFolder = strcat('InnerRace\',sel_diam);
    elseif strcmp(sel_fault,'BF')==1
        sel_file_names = T.File_Ball(sel_row);
        FaultTypeFolder = strcat('Ball\',sel_diam); 
    elseif strcmp(sel_fault,'OR_C6')==1 % OuterRace Centered @6
         sel_file_names = T.File_OR_C6(sel_row);
         FaultTypeFolder = strcat('OuterRace\centered6\',sel_diam); 
    elseif strcmp(sel_fault,'OR_O3')==1 % OuterRace Centered @3
         sel_file_names = T.File_OR_O3(sel_row);
         FaultTypeFolder = strcat('OuterRace\Ortogonal_3\',sel_diam);  
    elseif strcmp(sel_fault,'OR_OP12')==1 % OuterRace Centered OP12
         sel_file_names = T.File_OR_OP12(sel_row);
         FaultTypeFolder = strcat('OuterRace\Opposite_12\',sel_diam);      
    elseif strcmp(sel_fault,'Normal')==1   
        sel_file_names = T.Normal(sel_row);
        FaultTypeFolder = 'Normal\';
    end

%% read file name

    SaveFolder = "Signals_CSV/"; % 

    for j = 1:length(sel_file_names)
    
        sel_FaultDiam = T.FaultDiameter(sel_row(j));
        sel_file_name = sel_file_names{j};
        if strcmp(sel_file_name,"NA")==1
            continue
        end
        signals = load(strcat(FaultClassFolder,FaultTypeFolder,string(sel_file_name)));
        signal_names = fieldnames(signals);
        signal_values = struct2cell(signals);

        n_signals = length(signal_names);

        DE_signal = 'NA';
        FE_signal = 'NA';
        BA_signal = 'NA';
        RPM_nmb = -1;
        N_BA = 0; N_FE = N_BA; N_DE = N_FE;
    
        for i = 1:n_signals
            cur_name = signal_names{i,1};
    
            if contains(cur_name,"_DE_")
                DE_signal = signal_values{i,1};
                N_DE = length(DE_signal);
            elseif contains(cur_name,"_FE_")
                FE_signal = signal_values{i,1}; 
                N_FE = length(DE_signal);
            elseif contains(cur_name,"_BA_")
                BA_signal = signal_values{i,1};  
                N_BA = length(DE_signal);
            elseif contains(cur_name,"RPM")
                RPM_nmb = signal_values{i,1};
            end
        end

        if N_FE==0 || N_BA==0 || N_DE==0
            N_max = max([N_FE,N_DE,N_BA]);
       
            if N_DE==0
               DE_signal = zeros(N_max,1);  
            end
            if N_FE==0
                FE_signal = zeros(N_max,1);  
            end
            if N_BA==0
                BA_signal = zeros(N_max,1);  
            end
        end
%%
    FileName = "Signals_";
    FileName = strcat(FileName,Fault_SampRate, "_", sel_fault, "_", num2str(sel_FaultDiam),"_", num2str(sel_speed),".csv");


    T_write=table(DE_signal,FE_signal,BA_signal);
    writetable(T_write,strcat(SaveFolder,FileName),'WriteRowNames',true);


  end
end





