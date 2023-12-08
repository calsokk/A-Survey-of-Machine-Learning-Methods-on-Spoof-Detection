import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sp
from numpy import fft

# Take in data path
# Return [spoof list, bonafide list]
def create_bf_spf_lists(path):
    f = open(path)
    spoof_list = []
    bonafide_list = []
    for line in f:
        tokens = line.strip().split()
        if tokens[3] == 'spoof':
            spoof_list.append(tokens[1])
        else:
            bonafide_list.append(tokens[1])
    return [spoof_list, bonafide_list]

# Save Time Data to file
# take in spoof or bonafide list
# save time data to file
# num = number of samples to save
def extract_time_data(list, num):
    time_data = []
    #fspoof = open(path, "w")
    for i in range(num):
        dpath = 'DS_10283_3336\LA\LA\ASVspoof2019_LA_dev\\flac\\' + list[i] + '.flac'
        x, sr = librosa.load(dpath, sr=44100)
        time_data.append(x)
    return (time_data, sr)

def get_max_length(data):
    return max(len(vector) for vector in data)

def pad_or_truncate(data, max_length):
    padded_data = []
    for vector in data:
        if len(vector) < max_length:
            pad_width = max_length - len(vector)
            padded_vector = np.pad(vector, pad_width=(0, pad_width), mode='constant')
        elif len(vector) > max_length:
            padded_vector = vector[:max_length]
        else:
            padded_vector = vector
        padded_data.append(padded_vector)
    return padded_data


# Extract mfccs from time_data with padding
def extract_mfccs(time_data, sr=44100, n_mels=20, hop_length=8192):
    max_length = get_max_length(time_data)
    print(max_length)
    padded_data = pad_or_truncate(time_data, max_length)
    
    mfccs_data = []
    
    for x in padded_data:
        mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mfccs_data.append(mfccs)
        print(mfccs.shape)
    
    # Flatten to a large vector
    flattened_vector = np.array(mfccs).flatten()
    
    return stacked_dataset



'''
        mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mels=3)
        print(mfccs.shape)
        #plt.figure()
        #librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        #plt.show()
[spoof_list, bonafide_list] = create_bf_spf_lists("DS_10283_3336\LA\LA\ASVspoof2019_LA_asv_protocols\ASVspoof2019.LA.asv.dev.female.trl.txt")
save_time_data('female_time_spoof.txt', spoof_list, 1000)
save_time_data('female_time_bonafide.txt', bonafide_list, 1000)
'''

[spoof_list, bonafide_list] = create_bf_spf_lists("DS_10283_3336\LA\LA\ASVspoof2019_LA_asv_protocols\ASVspoof2019.LA.asv.dev.female.trl.txt")

time_data, sr = extract_time_data(spoof_list, 1000)
mfccs_data = extract_mfccs(time_data).T
print(mfccs_data.shape)

fspoof = open("spoof_data_mfccs1_male.txt", "w")
for row in mfccs_data:
    array_str = np.array_str(row, max_line_width=np.inf)              
    fspoof.write(array_str + '\n')

time_data, sr = extract_time_data(bonafide_list, 1000)
mfccs_data = extract_mfccs(time_data).T
print(mfccs_data.shape)

fbonafide = open("bonafide_data_mfccs1_male.txt", "w")
for row in mfccs_data:
    array_str = np.array_str(row, max_line_width=np.inf)              
    fbonafide.write(array_str + '\n')