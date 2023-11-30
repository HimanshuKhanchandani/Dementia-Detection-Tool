


import numpy as np
import matplotlib.pyplot as plt
import mne

from mne_connectivity import (spectral_connectivity_epochs,
                              spectral_connectivity_time)


from processing_functions import *


"""
###########################################################
###########################################################
Generate raw EEG data of desired epoch length.
###########################################################
###########################################################
"""


def generate_and_save_data(is_epoched,n_times,overlap_ratio,save = True, filename = "insert", subjects = [i for i in range(88)]):
    """
    Inputs:
    is_epoched: Boolean: should the data be epoched. (True = yes).
    n_times: Int: number of times in an epoch.
    overlap_ratio: Float: amount of overlap.
    save: Bool: save to file
    filename: string: filename of data. Assume it is run in same folder as 'processing_functions.py'
    total_subjects:
    
    Outputs:
    Saves np array:
    is_epoched = True: np array, shape = (n_epochs,n_times,n_channels)
    is_epoched = False: np array, shape = (n_times,n_channels) 
    
    """
    features = []
    for i in subjects:
        ppt = i + 1
        raw_data = mne.io.read_raw_eeglab('data/ds004504/derivatives/' + ppt_id(ppt)
                                    + '/eeg/' + ppt_id(ppt) + '_task-eyesclosed_eeg.set', preload = True)
        export = raw_data.to_data_frame()
        ppt_array = export.iloc[:,range(1,len(export.columns))].values
        
        # epoching data:
        if is_epoched == False:
            features += [ppt_array]
        else:
            ppt_array = create_epochs(ppt_array,n_times,overlap_ratio)
            features += [ppt_array]
    
    if save == True:
        np.save(filename, np.array(features, dtype=object), allow_pickle=True)
        return features
    else:
        return features
    


"""
###########################################################
###########################################################
Generate connectivity data of desired epoch length
and frequency bands
###########################################################
###########################################################
"""


def split_frequencies(freqs, f_bands, dictionary):
    """
    freqs: array of suitably spaced frequencies (np array)
    f_bands: array of strings (list); e.g., ["delta","alpha"]
    dictionary: dict object mapping band string to [fmin,fmax]; e.g., {"delta" => [0.5,4.0]}
    """
    F = [];
    for i in range(len(f_bands)):
        b = Freq_Bands[f_bands[i]];
        f = [j for j in freqs if (b[1] >= j >= b[0])]
        F += f
    return F



def subject_spectral_connectivity(data, channels, Freq_Bands, f_bands, f_spacing, sfreq, method, mode, f_avg = True, ch_types = "eeg"):
    '''
    INPUTS:
    data: np array: full raw eeg data
    channels: List: numeric list of channels.
    ch_names: List: corresponding list of channel names; e.g., "P2", etc.
    Freq_bands: dict: dictionary of band (string) and freq range; e.g., {"alpha" : [8.0,13.0]}.
    f_bands: List: list of frequency bands of interest; e.g., ["delta","alpha"].
    f_spacing: Int: Divide each 1 Hz band into N even pieces.
    sfreq: Float: sampling frequency
    method: string: connectivity measure to compute; ['coh', 'plv', 'ciplv', 'pli', 'wpli'].
    mode: string: method for estimating spectrum: 'multitaper', or 'cwt_morlet'.
    f_avg: Bool: Average over frequencies within desired bands.
    
    OUTPUTS:
    con_epochs_array: numpy array
    '''
    # subject data in shape (n_epochs,n_channels,n_times)
    data = np.swapaxes(data,1,2);
    
    if len(channels) == 1:
        channels = channels[0]
    # length of time segments
    n_times = data.shape[2];
    # number of channels:
    n_channels = len(channels);
    # number of epochs
    n_epochs = data.shape[0];
    
    # extract channels:
    data = data[:,channels,:];
    
    # create dictionary of channel names:
    channel_dict = {0: 'Fp1', 1: 'Fp2', 2: 'F3', 3: 'F4', 4: 'C3', 5: 'C4', 6: 'P3',
                    7: 'P4', 8: 'O1', 9: 'O2',10: 'F7', 11: 'F8', 12: 'T3', 13: 'T4',
                    14: 'T5', 15: 'T6', 16: 'Fz', 17: 'Cz', 18: 'Pz'}
    ch_names = [channel_dict[i] for i in channels]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    # create EpochsArray
    data_epoch = mne.EpochsArray(data, info)
    
    
    # frequency band information
    n_freq_bands = len(Freq_Bands)
    min_freq = np.min(list(Freq_Bands.values()))
    max_freq = np.max(list(Freq_Bands.values()))
    # Provide the freq points
    freqs_all = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))
    
    # split freq interval:
    freqs = split_frequencies(freqs_all, f_bands, Freq_Bands)
    
    # Tuples for desired bands
    fmin = tuple([Freq_Bands[f][0] for f in f_bands])
    fmax = tuple([Freq_Bands[f][1] for f in f_bands])
    
    con = spectral_connectivity_time(data_epoch, freqs = freqs, sfreq = sfreq,
                                      method=method,
                                      fmin=fmin, fmax=fmax, mode = mode,
                                      faverage=f_avg);
    
    # reshape data into (n_epochs,n_channels,n_channels,freqs)
    n_freqs = (con.get_data()).shape[2];
    con_data = con.get_data().reshape((n_epochs,n_channels,n_channels,n_freqs))
    
    return con_data



def connectivity_processing(epoched_data, channels, Freq_Bands, f_bands, f_spacing, sfreq, method, mode, subjects = [i for i in range(88)], f_avg = True, save = False, filename = "INSERT", ch_types = "eeg"):
    """
    NOTE: Assume input data is generated from function 'generate_and_save_data'
    
    INPUTS:
    epoched_data: np array (vector): array of np arrays.
    subjects: list: indices of all included subjects. Default all subjects = [1,...,88].
    channels: List: numeric list of channels.
    ch_names: List: corresponding list of channel names; e.g., "P2", etc.
    Freq_bands: dict: dictionary of band (string) and freq range; e.g., {"alpha" : [8.0,13.0]}.
    f_bands: List: list of frequency bands of interest; e.g., ["delta","alpha"].
    f_spacing: Int: Divide each 1 Hz band into N even pieces.
    sfreq: Float: sampling frequency
    method: string: connectivity measure to compute; ['coh', 'plv', 'ciplv', 'pli', 'wpli'].
    mode: string: method for estimating spectrum: 'multitaper', or 'cwt_morlet'.
    f_avg: Bool: Average over frequencies within desired bands.
    save: Bool: save data to file.
    filename: string.
    
    OUTPUTS:
    con_epochs_array: numpy array, shape = list(n_epochs, n_channels, n_channels, n_freq_bands)
    
    
    """
    output_array = []
    for idx in subjects:
        sub_data = all_data[idx];
        output = subject_spectral_connectivity(sub_data,channels, Freq_Bands, f_bands, f_spacing, sfreq, method, mode, f_avg = True, ch_types = "eeg")
        output_array += [output]
    
    
    if save == False:
        return output_array
    else:
        np.save(filename, np.array((np.array(output_array, dtype=object)), dtype=object), allow_pickle=True)
        return output_array
    

"""
###########################################################
###########################################################
Generate relative band power of desired epoch length.
###########################################################
###########################################################
"""    


def process(epoch_length,overlap_ratio,freq_bands,nperseg, absolute=False, sample_freq=500,total_subjects=88):
    features = []
    if absolute:
        absolute_features = []
        total_features = []
    for i in range(total_subjects):
        ppt = i + 1
        raw_data = mne.io.read_raw_eeglab('data/ds004504/derivatives/' + ppt_id(ppt)
                                    + '/eeg/' + ppt_id(ppt) + '_task-eyesclosed_eeg.set', preload = True)
        export = raw_data.to_data_frame()
        ppt_array = export.iloc[:,range(1,len(export.columns))].values
        ppt_epochs = create_epochs(ppt_array,epoch_length,overlap_ratio)
        freqs, ppt_psd  = welch(ppt_epochs,fs=sample_freq,nperseg=nperseg, axis=1)
        ppt_rbp = relative_band_power(ppt_psd,freqs,freq_bands)
        features += [ppt_rbp]
        if absolute:
            ppt_abp = absolute_band_power(ppt_psd,freqs,freq_bands)
            ppt_tbp = absolute_band_power(ppt_psd,freqs,[freq_bands[0],freq_bands[-1]])
            absolute_features += [ppt_abp]
            total_features += [ppt_tbp]
    num_epochs = np.array([feature.shape[0] for feature in features])
    """
    max_epoch_length = np.max(num_epochs)
    for i in range(total_subjects):
        shape = list(features[i].shape)
        shape[0] = max_epoch_length-shape[0]
        features[i] = np.concatenate((features[i],np.zeros(shape)),axis=0)
        if absolute:
            absolute_features[i] = np.concatenate((absolute_features[i],np.zeros(shape)),axis=0)
            total_shape = list(total_features[i].shape)
            total_shape[0] = max_epoch_length-total_shape[0]
            total_features[i] = np.concatenate((total_features[i],np.zeros(total_shape)),axis=0)
    """
    if not absolute:
        return features
    else:
        return num_epochs, np.array(features), np.array(absolute_features), np.array(total_features)


"""
###########################################################
###########################################################
Prepare connectivity data for training
###########################################################
###########################################################
"""


def reshape_sub_data(sub_data,epoch):
    # reshape sub_data into (1,number of connectivity features)
    sub_data = (sub_data[epoch].reshape(-1,))
    
    # remove zero elements:
    sub_data = [sub_data[i] for i in range(len(sub_data)) if sub_data[i] != 0]
    
    return np.array(sub_data).reshape((1,len(sub_data)))



def merge_sub_data(sub_data_list,epoch):
    N = len(sub_data_list)
    
    # initiate row of features:
    r = reshape_sub_data(sub_data_list[0],epoch)
    
    # concatenate all node/frequency combinations across axis = 1:
    for i in range(1,N):
        r_prime = reshape_sub_data(sub_data_list[i],epoch)
        r = np.concatenate((r,r_prime),axis = 1)
        
    return r



def get_con_data(con_data_lst,subjects):
    processed_con_data = [];
    K = len(con_data_lst)
    
    if type(subjects) == list:
        N = len(subjects)
    elif type(subjects) == int:
        N = subjects
    
    for i in range(N):
        # bin subjects data into list:
        sub_data_lst = [con_data_lst[j][i] for j in range(K)];
        
        # determine number of epochs
        n_epochs = sub_data_lst[0].shape[0];
        
        # if epochs are not the same length, return error:
        epochs_lst = [sub_data_lst[i].shape[0] for i in range(K)];
        epoch_bool = all(x == n_epochs for x in epochs_lst)
        
        if epoch_bool != True:
            raise ValueError('all epoch lengths must match.')
        else:
            
            # intialize np array:
            r = merge_sub_data(sub_data_lst,0)
            for j in range(1,n_epochs):
                r_prime = merge_sub_data(sub_data_lst,j);
                r = np.concatenate((r,r_prime),axis = 0)
            
            # add to list:
            processed_con_data += [r]
        
    return np.array(processed_con_data, dtype = object)

"""
###########################################################
###########################################################
Prepare relative band power features for training
###########################################################
###########################################################
"""

def reshape_rbp_data(data,freqs,nodes):
    data = data.copy()
    
    # remove freqs/nodes:
    f = [data[i][:,freqs,:][:,:,nodes] for i in range(len(data))]
    
    # reshape:
    f= [f[i].reshape((f[i].shape[0],f[i].shape[1]*f[i].shape[2])) for i in range(len(f))]
    
    return np.array(f, dtype = object)
    

"""
###########################################################
###########################################################
Merge connectivity and rbp features:
###########################################################
###########################################################
"""    


def get_con_rbp_data(con_data,rbp_data,subjects):
    """
    Note: assume that (1) con_data has been processed by above functions (2) rbp_data has also been processed
    by reshape_rbp_data.
    """
    
    if type(subjects) == list:
        N = len(subjects)
    elif type(subjects) == int:
        N = subjects
    
    processed_data = [];
    for i in range(N):
        # subject connectivity data:
        sub_con_data = con_data[i];
        # subject rbp data
        sub_rbp_data = rbp_data[i];
        # epoch length:
        n_epochs = sub_con_data.shape[0];
        
        # check if data matches:
        if n_epochs != sub_rbp_data.shape[0]:
            return "Number of epochs must match."
        else:
            # initialize np array
            sub_data = np.concatenate((sub_con_data,sub_rbp_data),axis = 1)
            
        # append to list:
        processed_data += [sub_data]
        
    return np.array(processed_data, dtype = object)


"""
###########################################################
###########################################################
Training functions:
###########################################################
###########################################################
"""   


def remove_test(features,test):
    features_train = [features[i] for i in range(len(features)) if i not in test]
    #target_train = [targets[i] for i in range(len(targets)) if i not in test]
    return features_train


def remove_test_targets(targets, test):
    target_train = [targets[i] for i in range(len(targets)) if i not in test]
    return target_train

def get_arr(data, idx , epoch): 
    trials = [(data[idx][epoch].reshape(-1,))[i] for i in range(16) if (data[idx][epoch].reshape(-1,))[i] != 0]
    m = len(trials)
    return (np.array(trials)).reshape((1,m))


def get_all_freq_arr(all_data,idx, epoch):
    r = get_arr(all_data[0], idx , epoch)
    for i in range(1,len(all_data)):
        r = np.concatenate((r,get_arr(all_data[i], idx , epoch)),axis =1)
        
    return r


def get_all_epochs_data(all_data,idx):
    r = get_all_freq_arr(all_data,idx,0)
    for j in range(1,len(all_data[0][idx])):
        r = np.concatenate((r,get_all_freq_arr(all_data,idx,j)),axis = 0)
    return r


def get_all_features(all_data):
    output_array = [];
    for j in range(len(all_data[0])):
        r = get_all_epochs_data(all_data,j)
        output_array = output_array + [r]
    return output_array


"""
###########################################################
###########################################################
Training functions:
###########################################################
###########################################################
"""  




def train_prep(features,targets,exclude=None,flatten_final=True):
    total_subjects = len(targets)
    target_list = []
    for i in range(total_subjects):
        num_epochs = features[i].shape[0]
        target_list.append(targets[i]*np.ones(num_epochs))
    if exclude==None: 
        features_array = np.concatenate(features)
        targets_array = np.concatenate(target_list)
    else:
        features_array = np.concatenate(features[:exclude] + features[exclude+1:])
        targets_array  = np.concatenate(target_list[:exclude] + target_list[exclude+1:])
    if flatten_final:
        features_array = features_array.reshape((features_array.shape[0],-1))
    return features_array, targets_array




def kNN_cross(rbps,targets,n_neighbors, PCA_components = 0):
   
    confusion_matrices_train = []
    confusion_matrices_test = []
    labels = np.unique(targets)
    for i in range(len(targets)):
        train_X, train_y = train_prep(rbps,targets,exclude=i,flatten_final=True)
        test_X = rbps[i].reshape(rbps[i].shape[0],-1)
        test_y = targets[i]*np.ones(rbps[i].shape[0])

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        
        if PCA_components != 0:
            pca = PCA(n_components = PCA_components)
            train_X = pca.fit_transform(train_X, y = None)
            test_X = pca.transform(test_X)
        
        ThreeNN = KNeighborsClassifier(n_neighbors=n_neighbors)
        ThreeNN.fit(train_X, train_y)
        
        confusion_matrices_train += [confusion_matrix(train_y, ThreeNN.predict(train_X),labels=labels)]
        confusion_matrices_test += [confusion_matrix(test_y,ThreeNN.predict(test_X),labels=labels)]
    
    confusion_matrices_train = np.array(confusion_matrices_train)
    confusion_matrices_test = np.array(confusion_matrices_test)
    total_confusion_train = np.sum(confusion_matrices_train, axis= 0)
    total_confusion_test = np.sum(confusion_matrices_test, axis= 0)
    
    train_metrics_dict = {'acc':accuracy(total_confusion_train), 'sens':sensitivity(total_confusion_train), 
                            'spec':specificity(total_confusion_train), 'f1':f1(total_confusion_train)}
    test_metrics_dict = {'acc':accuracy(total_confusion_test), 'sens':sensitivity(total_confusion_test), 
                            'spec':specificity(total_confusion_test), 'f1':f1(total_confusion_test)}
    
    
    return train_metrics_dict, test_metrics_dict


    




