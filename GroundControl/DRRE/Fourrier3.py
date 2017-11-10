#/usr/bin/python
#
# Function is called like:
# [ I, f, t ] = FourierAnalysis(datafile, time_window, recording_length, sampling_rate, radio_local_frequency, estimated_signal_frequency, estimated_signal_    width, buffer_size);
#
# Function needs to extract from raw recording the fourier transform around the expected signal.
#
# TBD: Joao's group had another technique that did not need the expected area, but could locate the signal with the raw file.
# TBD: this should be used to locate the file, which might be more robust.
#-------------------------------------------------------------------------------------------

# import other software 
import sys, getopt
import numpy as np
from scipy.fftpack import fft
import logging


def FourierAnalysis(datafile, time_window, recording_length, sampling_rate, radio_local_frequency, estimated_signal_frequency, estimated_signal_width,buffer_size):
  
  window_number_buffered = int(buffer_size / (time_window * sampling_rate) )*2
  if ( window_number_buffered < 1):
    window_number_buffered = 1
  counter = 0
  # Compute cut off bins of the central frequency +- half of the expected
  # frequency width plus 25 percent.
  min_bin = frequency_to_bin(estimated_signal_frequency - estimated_signal_width/2 - 0.25*estimated_signal_width, radio_local_frequency, sampling_rate, 1/time_window)
  max_bin = frequency_to_bin(estimated_signal_frequency + estimated_signal_width/2 + 0.25*estimated_signal_width, radio_local_frequency, sampling_rate, 1/time_window)
  # Initialise zero matrix to store Fourier values and counter at 1.
  fourier = np.zeros((int(recording_length/time_window), max_bin-min_bin+1))
#  try:
  with open(datafile,'rb') as f:
    for i in range(0, int(recording_length / (window_number_buffered * time_window))):
      data = np.fromfile(f, count=window_number_buffered*sampling_rate*time_window*2,dtype = np.float32)
      if data is None:
        break
      n = int(len(data)/2)

      # Reconstruct the complex array
      v = np.zeros([n], dtype = np.complex)
      v.real = data[::2]
      v.imag = -data[1::2]

      for i in range(window_number_buffered):
        if (counter < recording_length/time_window):
          subset_signal = v[(i*sampling_rate*time_window):((i+1)*sampling_rate*time_window)]
      #    break
          try:
            F = abs(fft(subset_signal, axis=0, overwrite_x=True))
          except(ValueError):
            break
          dumF = F
          
          # change spectrum such that it is correctly linked to
          # frequency values.
          F[0:int(len(F)/2)]  = dumF[int(len(F)/2):]
          F[int(len(F)/2):] = dumF[0:int(len(F)/2)]
          # Extract frequency zoom section.
          fourier[counter,:] = F[min_bin-1 : max_bin]
          counter += 1
      #break
  t = np.arange(0,((recording_length/time_window)-1)*time_window + time_window/2)
  f = bin_to_frequency(np.arange(min_bin,max_bin+1), estimated_signal_frequency, 1/sampling_rate, time_window)
  print('fourier', fourier.shape, f.shape, t.shape)
  return fourier, f, t

def bin_to_frequency(_bin, radio_frequency, sampling_frequency, frequency_step):
  return _bin * frequency_step + (radio_frequency - sampling_frequency/2) + frequency_step/2

def frequency_to_bin(frequency, radio_frequency, sampling_frequency, frequency_step):
  return int((frequency - (radio_frequency - sampling_frequency/2)) / frequency_step)

