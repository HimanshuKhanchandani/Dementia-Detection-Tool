import numpy as np

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
        indices.append(np.argmin(np.where(freqs >= freq_bands[i],freqs,np.inf)))
    return indices

def relative_band_power(ppt_psd,freqs,freq_bands):
    indices = freq_ind(freqs,freq_bands)
    total_power = np.sum(ppt_psd[:,indices[0]:indices[-1],:],axis=1,keepdims=True)
    relative_bands_list = []
    for i in range(len(indices)-1):
        relative_bands_list.append(np.sum(ppt_psd[:,indices[i]:indices[i+1],:],axis=1,keepdims=True)/total_power)
    return np.transpose(np.squeeze(np.array(relative_bands_list)),(1,0,2))

def create_numeric_labels(group_name):
    if group_name == 'A':
        return 0
    if group_name == 'F':
        return 1
    if group_name == 'C':
        return 2