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
from sympy import *
from mic_location import DoughertyLogSpiral, circular_3d_coords
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
        
def angle_3d(mic_loc, source_loc):
    pos = source_loc - mic_loc
    pos_x = [pos[0],pos[1], 0]
    
    azimuth_vec = [1,0,0]
    latitude_vec = [0,0,-1]
    
    pos_x_u = pos_x / np.linalg.norm(pos_x)
    pos_u = pos / np.linalg.norm(pos)
    # print(np.dot(pos_x_u, azimuth_vec))
    azimuth = np.arccos(np.dot(pos_x_u, azimuth_vec))
    latitude = np.arccos(np.dot(pos_u, latitude_vec))
    if pos[1] > 0:
        azimuth = np.pi - azimuth
    return azimuth*180.0/np.pi , latitude*180.0/np.pi 
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
    freq_range = [300, 4000]
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
    sig1_pos = [1, 1, 1.5]
    sig2_pos = [1, 5, 1.5]
    # noise_pos = [3,1,4]
    room.add_source(sig1_pos,signal=signal1,delay=0)
    room.add_source(sig2_pos,signal=signal2,delay=0) 
    # room.add_source(noise_pos,signal=noise,delay=0)
    # room.plot()
    # plt.show()
    mic_center = np.array([8, 3, 1])
    print('sig1_pos = ', angle_3d(sig1_pos, mic_center))
    print('sig2_pos = ', angle_3d(sig2_pos, mic_center))
    # print('noise_pos = ', angle_3d(mic_center, noise_pos))
    # microphone array radius
    mic_radius = 0.05
    # R = circular_3d_coords(mic_center, mic_radius, mic_n, 'vertical')
    R = DoughertyLogSpiral(mic_center, rmax = 0.15, rmin = 0.025, v = 87 *np.pi/180, n_mics = 112, direction = 'vertical')
    room.add_microphone_array(pra.MicrophoneArray(R, fs=room.fs))
    # room.compute_rir()
    room.simulate()
    room.plot()
    plt.show()
    X = pra.transform.stft.analysis(room.mic_array.signals.T, nfft, nfft // 2)
    # X = np.array([pra.transform.stft.analysis(signal, nfft, nfft // 2).T for signal in room.mic_array.signals])
    X = X.transpose([2, 1, 0])
    algo_names = ['MUSIC', 'TOPS']
    spatial_resp = dict()
    for algo_name in algo_names:
    # Construct the new DOA object
    # the max_four parameter is necessary for FRIDA only
        doa = pra.doa.algorithms[algo_name](R, Fs, nfft, c=c, num_src=2, max_four=4, dim = 3)
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
        print('azimuth = ', doa.azimuth_recon*180.0/np.pi)
        print('latitude = ', doa.colatitude_recon*180.0/np.pi)
        doa.grid.plot_old()
        plt.show()
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