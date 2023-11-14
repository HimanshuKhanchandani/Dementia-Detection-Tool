import numpy as np
import matplotlib.pyplot as plt
import mne

from mne_connectivity import (spectral_connectivity_epochs,
                              spectral_connectivity_time)

from processing_functions import create_epochs

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
    

# Generate Raw Data

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
    
