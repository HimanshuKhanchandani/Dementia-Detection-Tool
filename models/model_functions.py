import numpy as np
import mne
import config
from scipy.integrate import simpson

# test set
EXCLUDED = [4, 6, 8, 20, 33, 49, 53, 63, 71, 72]

def load_subject(id,path=config.PATH):
    """loads subject using their numeric id in the data folders"""
    return mne.io.read_raw_eeglab(path + '/derivatives/sub-' + str(id).zfill(3)
                                    + '/eeg/sub-' + str(id).zfill(3) + '_task-eyesclosed_eeg.set', preload = True,verbose='CRITICAL')

def subject_psd(raw,seg_length,fmin=0.5,fmax=45):
    """Computes the psd of each EEG channel for a given subject using Welch's method.

    Parameters
    ----------
    raw : 
        loaded RawEEGLAB data.
    seg_length : float
        length of each Welch segment in seconds. Determines frequency resolution. 
    fmin : float, optional
        Lower frequency of interest, by default 0.5.
    fmax : int, optional
        Upper frequency of interest, by default 45.

    Returns
    -------
    psd :
        psd of each EEG channel, stored in an mne Spectrum object.
    """        
    return raw.compute_psd(method='welch', fmin=fmin,fmax=fmax,n_fft=int(seg_length*raw.info['sfreq']),verbose=False)

def epochs_psd(raw,duration,overlap,seg_length,fmin=0.5,fmax=45,tmin=None,tmax=None):
    """Divides the EEG recording data into overlapping epochs for a given subject and 
    computes the psd of each EEG channel using Welch's method.

    Parameters
    ----------
    raw : 
        loaded RawEEGLAB data.
    duration : float
        Duration of each epoch in seconds.
    overlap : float
        overlap between epochs, in seconds.
    seg_length : float
        length of each Welch segment in seconds. Determines frequency resolution. 
    fmin : float, optional
        Lower frequency of interest, by default 0.5.
    fmax : int, optional
        Upper frequency of interest, by default 45.
    tmin : float, optional
        First time to include in seconds. Default value 'None' uses first time present in the data.
    tmax : float, optional
        Last time to include in seconds. Default value 'None' uses last time present in the data.

    Returns
    -------
    epoch_psds :
        psd of each EEG channel for each epoch, stored in an mne EpochsSpectrum object.
    """    
    epochs = mne.make_fixed_length_epochs(raw,duration=duration,preload=True,overlap=overlap,verbose=False)
    return epochs.compute_psd(method='welch', fmin=fmin,fmax=fmax,tmin=tmin,tmax=tmax,n_fft=int(seg_length*raw.info['sfreq']),verbose=False)

def freq_ind(freqs,freq_bands):
    """returns list of indices in the freqs array corresponding to the frequencies in freq_bands"""
    indices = []
    for i in range(len(freq_bands)):
        indices.append(np.argmin(np.abs(freqs-freq_bands[i])))
    return indices

def absolute_band_power(psds,freqs,freq_bands,endpoints=freq_ind):
    """Computes absolute band power in each frequency band of each EEG channel of row in the psds array.

    Parameters
    ----------
    psds : ndarray
        Array of psds of shape (num_rows) x (num_channels) x len(freqs)
    freqs : ndarray
        1-D array of frequencies.
    freq_bands : array_like
        List of frequencies defining the boundaries of the frequency bands.
    endpoints : 
        Function used to match freq_bands to freqs.

    Returns
    -------
    abps: ndarray
        Array of absolute band power values of shape (num_rows) x (num_channels) x (len(freq_bands)-1)
    """    
    indices = endpoints(freqs,freq_bands)
    absolute_bands_list = []
    for i in range(len(indices)-1):
        absolute_bands_list.append(simpson(psds[...,indices[i]:indices[i+1]+1],freqs[indices[i]:indices[i+1]+1],axis=-1))
    return np.transpose(np.array(absolute_bands_list),(1,2,0))

def relative_band_power(psds,freqs,freq_bands,endpoints=freq_ind):
    """Computes relative band power in each frequency band of each EEG channel of row in the psds array.

    Parameters
    ----------
    psds : ndarray
        Array of psds of shape (num_rows) x (num_channels) x len(freqs).
    freqs : ndarray
        1-D array of frequencies.
    freq_bands : array_like
        List of frequencies defining the boundaries of the frequency bands.
    endpoints : 
        Function used to match freq_bands to freqs.

    Returns
    -------
    rbps: ndarray
        Array of relative band power values of shape (num_rows) x (num_channels) x (len(freq_bands)-1).
    """    
    indices = endpoints(freqs,freq_bands)
    total_power = np.expand_dims(simpson(psds[...,indices[0]:indices[-1]+1],freqs[indices[0]:indices[-1]+1],axis=-1),axis=-1)
    return np.divide(absolute_band_power(psds,freqs,freq_bands,endpoints=endpoints),total_power)

def create_numeric_labels(group_name,labels={'A':1,'F':2,'C':0}):
    """assigns numeric labels to AD, FTD, and CN subjects based on labels dict"""
    return labels[group_name]




