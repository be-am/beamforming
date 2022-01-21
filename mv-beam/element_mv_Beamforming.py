from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
from scipy.io import wavfile
import soundfile as sf

fs, noise = wavfile.read("./input/male_voice.wav")
fs, signal = wavfile.read("./input/female_voice.wav")  # may spit out a warning when reading but it's alright!
signal = np.squeeze(signal[:,0])
noise = np.squeeze(noise[:,0])

sig_length = np.min([signal.shape[0], noise.shape[0]])
noise = noise[:sig_length] 
signal = signal[:sig_length]

# Create 4x6 shoebox room with source and interferer and simulate
room_mv_bf = pra.ShoeBox([4,6], fs=fs, max_order=0)
source = np.array([1, 4.5])
interferer = np.array([3.5, 3.])
room_mv_bf.add_source(source, delay=0., signal=signal)
room_mv_bf.add_source(interferer, delay=0., signal=noise)


center = [2, 1.5]; radius = 37.5e-3
fft_len = 1024
echo = pra.circular_2D_array(center=center, M=6, phi0=0, radius=radius)
echo = np.concatenate((echo, np.array(center, ndmin=2).T), axis=1)
mics = pra.Beamformer(echo, room_mv_bf.fs, N=fft_len)
room_mv_bf.add_microphone_array(mics)

mic_noise = 30
R_n = 10**((mic_noise-94)/20)*np.eye(fft_len*room_mv_bf.mic_array.M)
room_mv_bf.mic_array.rake_mvdr_filters(room_mv_bf.sources[1][:1], interferer = room_mv_bf.sources[0][:1], R_n = R_n)

fig, ax = room_mv_bf.plot(freq = [500, 1000, 2000 , 4000], img_order=0)
ax.legend(['500', '1000', '2000', '4000'])
fig.set_size_inches(20, 8)
ax.set_xlim([-3,8])
ax.set_ylim([-3,8])

room_mv_bf.simulate()
sf.write(r'C:\Users\bob04\pyroom\mv-beam\output_samples\all_mix.wav', room_mv_bf.mic_array.signals[-1,:],  fs)

#beamforming process
signal_mv = room_mv_bf.mic_array.process(FD=False)
sf.write(r'C:\Users\bob04\pyroom\mv-beam\output_samples\mv_beamforming.wav', signal_mv, fs)


