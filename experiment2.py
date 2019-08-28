import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
frequency_sampling, audio_signal = wavfile.read("C:/Users/Acer/Downloads/ENG_M.wav")
#display the parameters like sampling frequency of the audio signal, data type of signal and its duration
print('\nSignal shape:', audio_signal.shape)
print('Signal Datatype:', audio_signal.dtype)
print('Signal duration:', round(audio_signal.shape[0] /
float(frequency_sampling), 2), 'seconds')
audio_signal = audio_signal / np.power(2, 15) #normalization
#extracting the length and half length of the signal
length_signal = len(audio_signal)
half_length = np.ceil((length_signal + 1) / 2.0).astype(np.int)
signal_frequency = np.fft.fft(audio_signal) #transforming to frequency domain using furier transformation
#normalization of frequency domain signal and square it
t = np.arange(0, 10, 0.01);
a = np.sin(2*np.pi*t) + np.sin(2*2*np.pi*t) + np.sin(4*2*np.pi*t);
#itx = np.fft.ifft(signal_frequency)
signal_frequency = abs(signal_frequency[0:half_length]) / length_signal
signal_frequency **= 2
#extract the length and half length of the frequency transformed signal
len_fts = len(signal_frequency)
if length_signal % 2:
   signal_frequency[1:len_fts] *= 2
else:
   signal_frequency[1:len_fts-1] *= 2
   #extract the power in decibal(dB)
signal_power = 10 * np.log10(signal_frequency)
#Adjust the frequency in kHz for X-axis
x_axis = np.arange(0, half_length, 1) * (frequency_sampling / length_signal) / 1000.0
plt.figure()
plt.plot(x_axis, signal_power, color='black')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Signal power (dB)')
plt.show()
plt.plot(t, a);
plt.title("Time domain of the signal");
plt.xlabel('Time')
plt.ylabel('Amplitude')
#plt.grid(True)
plt.show();