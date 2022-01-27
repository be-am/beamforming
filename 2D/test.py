import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra

# center of array as column vector
mic_center = np.array([5, 5, 2])

# microphone array radius
mic_radius = 0.05

# number of elements
mic_n = 8

# Create the 2D circular points
R = pra.circular_2D_array(mic_center[:2], mic_n, 0, mic_radius)
R = np.concatenate((R, np.ones((1, mic_n)) * mic_center[2]), axis=0)

# Finally, we make the microphone array object as usual
# second argument is the sampling frequency
mics = pra.MicrophoneArray(R, 16000)

