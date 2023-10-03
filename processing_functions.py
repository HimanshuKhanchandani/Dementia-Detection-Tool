import numpy as np
from scipy.integrate import simpson
import mne
from scipy.signal import welch

def ppt_id(i):
    return 'sub-' + str(i).zfill(3)

def create_epochs(ppt_array,epoch_length,overlap_ratio):
    num_rows = ppt_array.shape[0]
    step_size = int((1-overlap_ratio)*epoch_length)
    epoch_list = []
    for i in range(int(num_rows/step_size)):
        if step_size*i+epoch_length <= num_rows: #leave out last epoch if it would have length less than epoch_length
            epoch_list.append(ppt_array[step_size*i:step_size*i+epoch_length])
    return np.array(epoch_list)

def freq_ind(freqs,freq_bands):
    indices = []
    for i in range(len(freq_bands)):
        indices.append(np.argmin(np.abs(freqs-freq_bands[i])))
    return indices

def relative_band_power(ppt_psd,freqs,freq_bands,endpoints=freq_ind):
    indices = endpoints(freqs,freq_bands)
    dx = freqs[1]-freqs[0]
    total_power = np.expand_dims(simpson(ppt_psd[:,indices[0]:indices[-1]+1,:],dx=dx,axis=1),axis=1)
    relative_bands_list = []
    for i in range(len(indices)-1):
        relative_bands_list.append(np.expand_dims(simpson(ppt_psd[:,indices[i]:indices[i+1]+1,:],dx=dx,axis=1),axis=1)/total_power)
    return np.transpose(np.squeeze(np.array(relative_bands_list)),(1,0,2))

def absolute_band_power(ppt_psd,freqs,freq_bands,endpoints=freq_ind):
    indices = endpoints(freqs,freq_bands)
    dx = freqs[1]-freqs[0]
    absolute_bands_list = []
    for i in range(len(indices)-1):
        absolute_bands_list.append(simpson(ppt_psd[:,indices[i]:indices[i+1]+1,:],dx=dx,axis=1))
    return np.transpose(np.array(absolute_bands_list),(1,0,2))

def create_numeric_labels(group_name):
    if group_name == 'A':
        return 1
    if group_name == 'F':
        return 2
    if group_name == 'C':
        return 0
        
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
    if not absolute:
        return num_epochs, np.array(features)
    else:
        return num_epochs, np.array(features), np.array(absolute_features), np.array(total_features)
    
def process_and_save(epoch_length,overlap_ratio,freq_bands,nperseg, filenames, absolute=False, sample_freq=500,total_subjects=88):
    if not absolute:
        num_epochs, rbp_features = process(epoch_length,overlap_ratio,freq_bands,nperseg, absolute=absolute, sample_freq=sample_freq,total_subjects=total_subjects)
        np.save(filenames[0],num_epochs)
        np.save(filenames[1],rbp_features)
    else:
        num_epochs, rbp_features, abp_features, tbp_features = process(epoch_length,overlap_ratio,freq_bands,nperseg, absolute=absolute, sample_freq=sample_freq,total_subjects=total_subjects)
        np.save(filenames[0],num_epochs)
        np.save(filenames[1],rbp_features)
        np.save(filenames[2],abp_features)
        np.save(filenames[3],tbp_features)