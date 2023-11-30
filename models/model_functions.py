import numpy as np
import pandas as pd
import mne
from config import DATA_PATH, PROCESSED_DATA_PATH
import pickle
from scipy.integrate import simpson

# STABLE

'''
This python file contains all the necessary functions to take the data and extract the relative band power features we use for modeling. The process involves taking the EEG and chopping it up into epochs, calculating relative band powers and then saving the relative band powers as a pickle file. The file also contains some functions necessary to prepare the data for the training algorithm.
'''

# subject ids of test set 
TEST = [4, 6, 8, 20, 33, 49, 53, 63, 71, 72]

# Dictionaries mapping the second-to-last array index in the output of load_data to the corresponding EEG channel. 
CHANNELS = {0: 'Fp1', 1: 'Fp2', 2: 'F3', 3: 'F4', 4: 'C3', 5: 'C4', 6: 'P3', 7: 'P4', 8: 'O1', 9: 'O2', 
 10: 'F7', 11: 'F8', 12: 'T3', 13: 'T4', 14: 'T5', 15: 'T6', 16: 'Fz', 17: 'Cz', 18: 'Pz'}

channels_dict = {'Fp1' : 0, 'Fp2': 1 , 'F3': 2, 'F4' : 3, 'C3': 4, 'C4': 5, 'P3': 6, 'P4' : 7, 'O1' : 8, 'O2':9, 
'F7':10 , 'F8': 11, 'T3' : 12, 'T4' :13, 'T5' : 14, 'T6' : 15, 'Fz': 16, 'Cz': 17,'Pz': 18}


def load_subject(subject_id,path=DATA_PATH):
    """loads subject using their numeric id in the data folders"""
    return mne.io.read_raw_eeglab(path + '/derivatives/sub-' + str(subject_id).zfill(3)
                                    + '/eeg/sub-' + str(subject_id).zfill(3) + '_task-eyesclosed_eeg.set', preload = True,verbose='CRITICAL')

def subject_psd(raw,seg_length,fmin=0.5,fmax=45):
    """
    Computes the psd of each EEG channel for a given subject using Welch's method.

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
    """
    Divides the EEG recording data into overlapping epochs for a given subject and 
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
    epochs_psds :
        psd of each EEG channel for each epoch, stored in an mne EpochsSpectrum object.
    """    
    epochs = mne.make_fixed_length_epochs(raw,duration=duration,preload=True,overlap=overlap,verbose=False)
    return epochs.compute_psd(method='welch', fmin=fmin,fmax=fmax,tmin=tmin,tmax=tmax,n_fft=int(seg_length*raw.info['sfreq']),verbose=False)

def load_data(duration,overlap,seg_length,fmin=0.5,fmax=45,classes={'A':1,'F':2,'C':0},path=DATA_PATH):
    """
    Loads all subjects from the specified classes, divides their EEG recordings into epochs, 
    computes the psd of each EEG channel using Welch's method, and then returns those psds with 
    the assigned class labels.

    Parameters
    ----------
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
    classes : dict, optional
        Dictionary whose keys are the classes to include and values are the numeric labels. 
        By default {'A':1,'F':2,'C':0}.
    path : str, optional
        filepath to data folder. Defaults to PATH in config.py

    Returns
    -------
    subject_data : list[ndarray]
        List of arrays, each of shape (num_epochs) x (num_channels) x len(freqs), 
        with each array corresponding to a subject.
    freqs : ndarray
        Array of frequencies at which the psds are measured.
    targets : ndarray
        Numeric class labels for the subjects in subject_data. 
    """    
    subject_table = pd.read_csv(path + '/participants.tsv',sep='\t')
    target_labels = subject_table['Group']
    subject_data = []
    targets = []
    for subject_id in range(1,len(target_labels)+1):
        if target_labels.iloc[subject_id-1] not in classes:
            continue
        raw = load_subject(subject_id,path=path)
        epochs_psds = epochs_psd(raw,duration=duration,overlap=overlap,seg_length=seg_length,fmin=fmin,fmax=fmax)
        epochs_psds_array, freqs = epochs_psds.get_data(return_freqs=True)
        subject_data.append(epochs_psds_array)
        targets.append(classes[target_labels.iloc[subject_id-1]])
    return subject_data, freqs, np.array(targets)

def save_psds(subject_data,freqs,targets,filename,path=PROCESSED_DATA_PATH):
    """Pickles the psd data generated by load_data and saves it in the data processing folder"""
    with open(path + '/' + filename,'wb') as file:
        pickle.dump({'subject_data':subject_data,'freqs':freqs,'targets':targets},file)

def load_psds(filename,path=PROCESSED_DATA_PATH):
    """Loads pickled psd data from the data processing folder"""
    with open(path + '/' + filename,'rb') as file:
        psds = pickle.load(file)
    return psds['subject_data'], psds['freqs'], psds['targets']

def freq_ind(freqs,freq_bands):
    """returns list of indices in the freqs array corresponding to the frequencies in freq_bands"""
    indices = []
    for i in range(len(freq_bands)):
        indices.append(np.argmin(np.abs(freqs-freq_bands[i])))
    return indices

def absolute_band_power(psds,freqs,freq_bands,endpoints=freq_ind):
    """
    Computes absolute band power in each frequency band of each EEG channel of row in the psds array.

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
    """
    Computes relative band power in each frequency band of each EEG channel of row in the psds array.

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

def align_test_labels(test=TEST,classes=['A','C','F']):
    """
    Aligns test set labels based on the classes that a classifier is training on. Only supports orders
    ['A','C','F'], ['A','C'], ['C','F'], and ['A','F'].
    """
    if classes == ['A','C','F']:
        return [subject_id-1 for subject_id in test]
    if classes == ['A','C']:
        return [subject_id-1 for subject_id in test if subject_id <= 65]
    if classes == ['C','F']:
        return [subject_id-37 for subject_id in test if subject_id >= 37]
    if classes == ['A','F']:
        return ([subject_id-1 for subject_id in test if subject_id <= 36] 
                + [subject_id-30 for subject_id in test if subject_id >= 66])

def remove_class(features,targets,class_):
    """
    Removes a class from the loaded data. Used when all three classes are loaded but only two
    are being used for modeling. 
    """
    if class_ == 'F':
        return features[:65],targets[:65]
    if class_ == 'A':
        return features[36:],targets[36:]
    if class_ == 'C':
        return features[:36]+features[65:], np.append(targets[:36], targets[65:])

def remove_test(features,targets,test):
    """
    Removes test subjects from the list of feature arrays. Before using this function the labels
    should be aligned with align_test_labels first based on the classification problem under consideration.
    """
    features_train = [features[i] for i in range(len(features)) if i not in test]
    target_train = [targets[i] for i in range(len(targets)) if i not in test]
    return features_train, target_train

def select_test(features,targets,test):
    """
    Selects test subjects from the list of feature arrays. Before using this function the labels
    should be aligned with align_test_labels first based on the classification problem under consideration.
    """
    features_train = [features[i] for i in range(len(features)) if i in test]
    target_train = [targets[i] for i in range(len(targets)) if i in test]
    return features_train, target_train

def remove_channel(input_rbp, channels_to_remove):
    """
    removes a list of EEG channels from the input list of feature arrays containing all 19 EEG channels.
    
    Parameters
    ----------
    input_rbp : list[ndarray]
            List of feature arrays corresponding to each subject containing all 19 channels. The channel corresponds to 
            second-to-last array index for the arrays.
    channels_to_remove: list
            List of EEG channels to remove.
    
    Returns
    -------
    updated_rbp: list[ndarray]
            List of feature arrays corresponding to each subject not containing the removed channels.
    """
    
    updated_rbp = []
    all_channels = np.arange(0,19)
    channels_removed_ind = [channels_dict[ch] for ch in channels_to_remove]
    resulting_channels = np.delete(all_channels, channels_removed_ind)
    updated_rbp = [rbp[:, resulting_channels, :] for rbp in input_rbp]
    return updated_rbp



def train_prep(features,targets,exclude=None,flatten_final=True):
    """
    Prepares a list of feature arrays with corresponding labels in targets for training by concatenating
    along the first (epochs) dimension. Optionally excludes a subject for leave-one-subject-out
    cross-validation

    Parameters
    ----------
    features : list[ndarray]
        List of feature arrays corresponding to each subject. 
    targets : ndarray
        Array of numeric class labels for each feature array.
    exclude : int, optional
        Excludes the feature array of the subject with index 'exclude' from the output. 
        The default value of 'None' keeps all subjects in. 
    flatten_final : bool, optional
        The default value True flattens all dimensions of the output feature array except the first dimension.
        Setting this to False preserves the features as a 2d array of shape num_channels*num_frequency_bands. 

    Returns
    -------
    features_array : ndarray
        Array of features with each row corresponding to a training example.
    targets_array : ndarray
        1-D array of labels for the training examples in features_array.
    """
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

def accuracy(confusion):
    """Calculates accuracy from a confusion matrix."""
    return np.trace(confusion)/np.sum(confusion)
def sensitivity(confusion):
    """
    Calculates sensitivity from a 2 x 2 confusion matrix.
    Index 0 corresponds to negative examples and index 1 corresponds to positive examples.
    """
    return confusion[1,1]/(confusion[1,1]+confusion[1,0])
def specificity(confusion):
    """
    Calculates specificity from a 2 x 2 confusion matrix.
    Index 0 corresponds to negative examples and index 1 corresponds to positive examples.
    """
    return confusion[0,0]/(confusion[0,0]+confusion[0,1])
def precision(confusion):
    """
    Calculates precision from a 2 x 2 confusion matrix.
    Index 0 corresponds to negative examples and index 1 corresponds to positive examples.
    """
    return confusion[1,1]/(confusion[1,1]+confusion[0,1])
def f1(confusion):
    """
    Calculates F1 score for a 2 x 2 confusion matrix.
    Index 0 corresponds to negative examples and index 1 corresponds to positive examples.
    """
    return 2*(precision(confusion)*sensitivity(confusion))/(precision(confusion)+sensitivity(confusion))




