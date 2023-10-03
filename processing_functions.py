import numpy as np
from scipy.integrate import simpson

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