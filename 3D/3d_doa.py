from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra
# from pyroomacoustics.directivities import DirectivityPattern, DirectionVector, CardioidFamily
import soundfile as sf
import samplerate
from scipy import signal
from scipy.signal import butter, lfilter
import os
import librosa
import librosa.display
import pyroomacoustics.directivities


def preprocess_data(signal, Fs):
    np.array(signal, dtype=float)
    signal = pra.highpass(signal, Fs)
    signal = pra.normalize(signal)

    if np.shape(signal)[1]:
        signal = np.squeeze(signal[:,0])
    return signal

def resample(ori_rate,new_rate,signal):
    fs_ratio = new_rate / float(ori_rate)
    signal = samplerate.resample(signal, fs_ratio, "sinc_best")
    return signal


def circular_3d_coords(center, radius, num, direction = 'virtical'):

    list_coords = []

    if direction == 'vertical':
        for i in range(num):
            list_coords.append([center[0], center[1] + radius*np.sin(2*i*np.pi/num), center[2] + radius*np.cos(2*i*np.pi/num)])

    elif direction == 'horizontal':
        for i in range(num):
            list_coords.append([center[0]+ radius*np.sin(2*i*np.pi/num), center[1]+ radius*np.cos(2*i*np.pi/num), center[2] ])
    list_coords = [list(reversed(col)) for col in zip(*list_coords)]

    return np.array(list_coords)
        
def angle_3d(mic_loc, source_loc):
    pos = source_loc - mic_loc
    azimuth_vec = [0,0,1]
    latitude_vec = [pos[0],0,pos[2]]

    pos_u = pos / np.linalg.norm(pos)
    azimuth_vec_u = azimuth_vec / np.linalg.norm(azimuth_vec)
    latitude_vec_u = latitude_vec / np.linalg.norm(latitude_vec)

    azimuth = np.arccos(np.clip(np.dot(pos_u, azimuth_vec_u), -1.0, 1.0))
    latitude = np.arccos(np.clip(np.dot(pos_u, latitude_vec_u), -1.0, 1.0))


    return azimuth, latitude



if __name__ == "__main__":

    '''
        s1, s1 ,s2 ,n1  
        sum + rir 
        rakeMVDR, delay and sum, rake pertual MVDR  -- 스펙트로그램까지 
    '''
    # Spectrogram figure properties
    figsize = (15, 7)  # figure size
    fft_size = 512  # fft size for analysis
    fft_hop = 8  # hop between analysis frame
    fft_zp = 512  # zero padding
    analysis_window = pra.hann(fft_size)
    t_cut = 0.83  # length in [s] to remove at end of signal (no sound)    


    # Some simulation parameters
    Fs = 8000
    absorption = 0.1
    max_order_sim = 0
    sigma2_n = 5e-7
    c = 343.
    freq_range = [300, 3500]

    # Microphone array design parameters
    mic_n = 8  # number of microphones
    d = 0.08  # distance between microphones
    phi = 0.0  # angle from horizontal
    max_order_design = 1  # maximum image generation used in design
    shape = "Linear"  # array shape
    Lg_t = 0.100  # Filter size in seconds
    Lg = np.ceil(Lg_t * Fs)  # Filter size in samples
    delay = 0.050  # Beamformer delay in seconds

    # Define the FFT length
    N = 1024
    nfft = 256

    #Define two signal and one noise
    path = os.path.dirname(__file__) 

    rate1, signal1 = wavfile.read(path + "/input_samples/female_voice.wav")  # may spit out a warning when reading but it's alright!
    signal1 = preprocess_data(signal1, Fs)

    rate2, signal2 = wavfile.read(path + "/input_samples/male_voice.wav")
    signal2 = preprocess_data(signal2, Fs)

    rate3, noise = wavfile.read(path + "/input_samples/cafe.wav")
    noise = preprocess_data(noise, Fs)

    #resample audio file for same Fs
    new_rate = 8000
    signal1 = resample(rate1,new_rate,signal1)
    signal2 = resample(rate2,new_rate,signal2)
    noise = resample(rate3,new_rate,noise)

    sig_length = np.min([signal1.shape[0], signal2.shape[0], noise.shape[0]])
    signal1 = signal1[:sig_length]
    signal2 = signal2[:sig_length]
    noise = noise[:sig_length]

    
    # Create a 10X5X5 metres shoe box room
    room_dim=[10,5,5]
    room = pra.ShoeBox(
        room_dim,
        absorption=absorption,
        fs=Fs,
        max_order=max_order_sim,
        sigma2_awgn=sigma2_n,
    )

    sig1_pos = [1,2,1.5]
    sig2_pos = [1,4,1.5]
    noise_pos = [3,1,4]

    room.add_source(sig1_pos,signal=signal1,delay=0)
    room.add_source(sig2_pos,signal=signal2,delay=0) 
    room.add_source(noise_pos,signal=noise,delay=0)

    mic_center = np.array([8, 3, 1])

    print('sig1_pos = ', angle_3d(mic_center, sig1_pos))
    print('sig2_pos = ', angle_3d(mic_center, sig2_pos))
    print('noise_pos = ', angle_3d(mic_center, noise_pos))
    # microphone array radius
    mic_radius = 0.05
    R = circular_3d_coords(mic_center, mic_radius, mic_n, 'vertical')
    room.add_microphone_array(pra.MicrophoneArray(R, fs=room.fs))

    room.compute_rir()
    room.simulate()

    X = pra.transform.stft.analysis(room.mic_array.signals.T, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])

    algo_names = ['SRP', 'MUSIC', 'TOPS']
    spatial_resp = dict()

    for algo_name in algo_names:
    # Construct the new DOA object
    # the max_four parameter is necessary for FRIDA only
        doa = pra.doa.algorithms[algo_name](R, Fs, nfft, c=c, num_src=3, max_four=4, dim = 3)

        # this call here perform localization on the frames in X
        doa.locate_sources(X, freq_range=freq_range)
        
        # store spatial response
        if algo_name is 'FRIDA':
            spatial_resp[algo_name] = np.abs(doa._gen_dirty_img())
        else:
            spatial_resp[algo_name] = doa.grid.values
            
        # normalize   
        min_val = spatial_resp[algo_name].min()
        max_val = spatial_resp[algo_name].max()
        spatial_resp[algo_name] = (spatial_resp[algo_name] - min_val) / (max_val - min_val)
        print('algorithm = ', algo_name)
        print('azimuth = ', doa.azimuth_recon)
        print('latitude = ', doa.colatitude_recon)


    # Design the beamforming filters using some of the images sources

    
    # wavfile.write(
    #     path + "/output_samples/output_PerceptualMvdr_45ms.wav", Fs, out_RakePerceptual.astype(np.float32)
    # )


    # room.plot(freq=[7000],img_order=0)
    # plt.show()
    
    # dSNR = pra.dB(room.direct_snr(mics.center[:, 0], source=0), power=True)
    # print("The direct SNR for good source is " + str(dSNR))

    # S = librosa.feature.melspectrogram(y=out_RakePerceptual, sr=Fs,n_fft=fft_size,hop_length=fft_hop, n_mels=128,window=analysis_window) 
    
    # log_S = librosa.amplitude_to_db(S, ref=np.max)
    # plt.figure(figsize=(12, 4))
    # librosa.display.specshow(log_S, sr=Fs, x_axis='time', y_axis='mel')
    # plt.title('mel power spectrogram')
    # plt.colorbar(format='%+02.0f dB')
    # plt.tight_layout()
    # plt.savefig(path + "/output_samples/spectrograms_PerceptualMvdr_45ms.png", dpi=150)
    # plt.show()
