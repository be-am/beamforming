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
        



if __name__ == "__main__":

    '''
        s1, s1 ,s2 ,n1  
        sum + rir 
        rakeMVDR, delay and sum, rake pertual MVDR  -- 스펙트로그램까지 
    '''

    # Some simulation parameters
    Fs = 8000
    absorption = 0.1
    max_order_sim = 1
    sigma2_n = 5e-7

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

    # Add two source somewhere in the room
    room.add_source([1,3,1.5],signal=signal1,delay=0)
    room.add_source([1,4,1.5],signal=signal2,delay=0) 
    room.add_source([3,1,4],signal=noise,delay=0)


    # center of array as column vector
    mic_center = np.array([8, 3, 1])
    # microphone array radius
    mic_radius = 0.05
    # Create the 2D circular points
    # R = pra.circular_2D_array(mic_center[:2], mic_n, phi, mic_radius)
    # R = np.concatenate((R, np.ones((1, mic_n)) * mic_center[2]), axis=0)

    R = circular_3d_coords(mic_center, mic_radius, mic_n, 'vertical')

    # Finally, we make the microphone array object as usual
    # second argument is the sampling frequency    

    """
    Rake Perceptual simulation
    """

    # compute beamforming filters
    mics = pra.Beamformer(R, Fs, N, Lg=Lg)
    room.add_microphone_array(mics)
    

    fig, ax = room.plot(freq=[500, 1000, 2000, 4000], img_order=0)
    # ax.legend(['500', '1000', '2000', '4000'])

    plt.show()

    room.mic_array = mics
    room.compute_rir()
    room.simulate()

    # Design the beamforming filters using some of the images sources
    good_sources = room.sources[0][: max_order_design + 1]
    bad_sources1 = room.sources[2][: max_order_design + 1]
    
    mics.rake_perceptual_filters(
        good_sources, bad_sources1, sigma2_n * np.eye(mics.Lg * mics.M), delay=0
    )

    # process the signal
    output = mics.process()

    # save to output file
    out_RakePerceptual = pra.normalize(pra.highpass(output, Fs))
    
    wavfile.write(path + "/output_samples/all_mix.wav", Fs, room.mic_array.signals[-1,:].astype(np.float32))
    wavfile.write(
        path + "/output_samples/output_RakePerceptual.wav", Fs, out_RakePerceptual.astype(np.float32)
    )

    room.plot(freq=[6000],img_order=0)
    plt.show()


