import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sp
from numpy import fft


NUM_BSAMP = 1500
NUM_SSAMP = 4500

fspoof = open("spoof_data_freq_female.txt", "w")
fbonafide = open("bonafide_data_freq_female.txt", "w")
# Process female eval data
ffemale = open("DS_10283_3336\LA\LA\ASVspoof2019_LA_asv_protocols\ASVspoof2019.LA.asv.eval.female.trl.txt")

spoof_list = []
bonafide_list = []
for line in ffemale:
    tokens = line.strip().split()
    if tokens[3] == 'spoof':
        spoof_list.append(tokens[1])
    else:
        bonafide_list.append(tokens[1])

for i in range(NUM_BSAMP):
    #path = 'DS_10283_3336\LA\LA\ASVspoof2019_LA_dev\\flac\\' + bonafide_list[i] + '.flac'
    #for eval:
    path = 'DS_10283_3336\LA\LA\ASVspoof2019_LA_eval\\flac\\' + bonafide_list[i] + '.flac'
    data, samplerate = sf.read(path)
    print(samplerate)
    data_fft = sp.fftshift(sp.fft(data))       
    len_fft = len(data_fft)                 
    # TEMP SOLUTION TO GET VECTORS SAME - INACCURATE                
    sampled_fft = np.abs(data_fft[range(len_fft//2,len_fft,len_fft//120)]) 
    if (len(sampled_fft) == 61):
        sampled_fft = sampled_fft[:-1]
    array_str = np.array_str(sampled_fft, max_line_width=np.inf)              
    fbonafide.write(array_str + '\n')

for i in range(NUM_SSAMP):
    #path = 'DS_10283_3336\LA\LA\ASVspoof2019_LA_dev\\flac\\' + spoof_list[i] + '.flac'
    #for eval
    path = 'DS_10283_3336\LA\LA\ASVspoof2019_LA_eval\\flac\\' + spoof_list[i] + '.flac'
    data, samplerate = sf.read(path)
    data_fft = sp.fftshift(sp.fft(data))       
    len_fft = len(data_fft)
    #print(len_fft) 
    # TEMP SOLUTION TO GET VECTORS SAME - INACCURATE                
    sampled_fft = np.abs(data_fft[range(len_fft//2,len_fft,len_fft//120)]) 
    #print(len(sampled_fft))  
    if (len(sampled_fft) == 61):
        sampled_fft = sampled_fft[:-1]
    array_str = np.array_str(sampled_fft, max_line_width=np.inf)              
    fspoof.write(array_str + '\n')
    if len(sampled_fft) == 61:
        print('dum')


'''
path = 'DS_10283_3336\LA\LA\ASVspoof2019_LA_dev\\flac\\' + bonafide_list[0] + '.flac'
data, samplerate = sf.read(path) 
print(samplerate)                                           
dataset = [data, data]
lendata = len(data)

data_fft = np.fft.fft(data)
n = data.size
timestep = 1/16000
freq_vals = np.fft.fftfreq(n, d=timestep)

len_fft = len(data_fft)
sampled_fft = data_fft[range(len_fft//2,len_fft,500)]
print(len(sampled_fft))

time_vals = np.array(range(0, len(data))) * 1/16000
plt.plot(time_vals, data)
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.show()
plt.plot(freq_vals, data_fft)
plt.xlabel('frequency(Hz)')
plt.ylabel('fourier tranform amplitude')
plt.show()
plt.plot(freq_vals[-1 * np.array(range(len_fft//2,len_fft,500))], sampled_fft)
plt.show()
'''

fspoof = open("spoof_data_freq_male.txt", "w")
fbonafide = open("bonafide_data_freq_male.txt", "w")

# Process MALE FREQ data
fmale = open("DS_10283_3336\LA\LA\ASVspoof2019_LA_asv_protocols\ASVspoof2019.LA.asv.eval.male.trl.txt")
spoof_list = []
bonafide_list = []
for line in fmale:
    tokens = line.strip().split()
    if tokens[3] == 'spoof':
        spoof_list.append(tokens[1])
    else:
        bonafide_list.append(tokens[1])

for i in range(NUM_BSAMP):
    #path = 'DS_10283_3336\LA\LA\ASVspoof2019_LA_dev\\flac\\' + bonafide_list[i] + '.flac'
    #for eval
    path = 'DS_10283_3336\LA\LA\ASVspoof2019_LA_eval\\flac\\' + bonafide_list[i] + '.flac'
    data, samplerate = sf.read(path)
    data_fft = sp.fftshift(sp.fft(data))       
    len_fft = len(data_fft)                 
    # TEMP SOLUTION TO GET VECTORS SAME - INACCURATE                
    sampled_fft = np.abs(data_fft[range(len_fft//2,len_fft,len_fft//120)]) 
    if (len(sampled_fft) == 61):
        sampled_fft = sampled_fft[:-1]
    array_str = np.array_str(sampled_fft, max_line_width=np.inf)              
    fbonafide.write(array_str + '\n')
    if len(sampled_fft) == 61:
        print('dum')

for i in range(NUM_SSAMP):
    #path = 'DS_10283_3336\LA\LA\ASVspoof2019_LA_dev\\flac\\' + bonafide_list[i] + '.flac'
    #for eval
    path = 'DS_10283_3336\LA\LA\ASVspoof2019_LA_eval\\flac\\' + bonafide_list[i] + '.flac'
    data, samplerate = sf.read(path)
    data_fft = sp.fftshift(sp.fft(data))       
    len_fft = len(data_fft)
    #print(len_fft) 
    # TEMP SOLUTION TO GET VECTORS SAME - INACCURATE                
    sampled_fft = np.abs(data_fft[range(len_fft//2,len_fft,len_fft//120)]) 
    #print(len(sampled_fft))  
    if (len(sampled_fft) == 61):
        sampled_fft = sampled_fft[:-1]
    array_str = np.array_str(sampled_fft, max_line_width=np.inf)              
    fspoof.write(array_str + '\n')
    if len(sampled_fft) == 61:
        print('dum')