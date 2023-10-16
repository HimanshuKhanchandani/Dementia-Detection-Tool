import numpy as np
import mne

def create_numeric_labels(group_name,labels={'A':1,'F':2,'C':0}):
    # assigns numeric labels to AD, FTD, and CN subjects based on labels dict
    return labels[group_name]

def load_subject(id):
    # loads subject using their numeric id in the data folders
    # returns a Raw object
    return mne.io.read_raw_eeglab('../data/ds004504/derivatives/sub-' + str(id).zfill(3)
                                    + '/eeg/sub-' + str(id).zfill(3) + '_task-eyesclosed_eeg.set', preload = True,verbose='CRITICAL')
    
def freq_ind(freqs,freq_bands):
    indices = []
    for i in range(len(freq_bands)):
        indices.append(np.argmin(np.abs(freqs-freq_bands[i])))
    return indices
