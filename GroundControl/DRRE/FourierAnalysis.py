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

# Start function

def FourierAnalysis(datafile, time_window, recording_length, sampling_rate, radio_local_frequency, estimated_signal_frequency, estimated_signal_width,buffer_size):

   # input filename
   inputfilename = datafile

   # initail values
   Time = recording_length
   Fs = sampling_rate
   Dt = time_window

   # Construct other information
   L = Time*Fs
   setWav = int(2*L/(Time/Dt))

   # get optimal NFFT value
   #n2 = 1 #nextpow 2 algoritm
   #while n2 < L/Time/Dt: n2 *= 2
   #NFFT = 2^n2 
   NFFT = 65536.
   #half = NFFT/2

   # get values for axis waterfall plot
   f0 = float(radio_local_frequency)
   nx = float(Fs)/float(NFFT)
   ny = float(Dt)
   bandwidth = np.arange(f0-Fs/2+nx/2,f0+Fs/2-nx/2+1,nx)
   time = np.arange(1,Time+1,ny)

   # set certain zoom area
   lbound = float(estimated_signal_frequency) - float(estimated_signal_width)
   rbound = float(estimated_signal_frequency) + float(estimated_signal_width)

   LF = np.array((bandwidth>lbound).nonzero())
   lfreq = LF.min()+1
   RF = np.array((bandwidth>rbound).nonzero())
   rfreq = RF.min()+1

   tmp = bandwidth[lfreq:rfreq+1]
   numC = tmp.size

   I = np.zeros([Time/Dt,numC],dtype='float')

   # Initialize Matrices
   forend = int(Time/Dt)
   f = open(inputfilename,'rb')
   for i in range(0,forend):
        # start reading the file
        t = np.fromfile(f,dtype = np.float32, count = setWav)
        n = len(t)/2

        # Reconstruct the complex array
        v = np.zeros([n], dtype = np.complex)
        v.real = t[::2]
        v.imag = -t[1::2]
        
        # generate the fft of the signal
        Y = fft(v,NFFT)
        dum = 2*abs(Y[0:NFFT+1]/setWav)
        half = len(dum)/2

        # fill in the Waterfall matrix
        line = np.zeros(dum.shape)
        line[0:half] = dum[half:]
        line[half:] = dum[0:half]
        line = line.conj().transpose()
        #line = line[::-1]

        invec = line[lfreq:rfreq+1]
        I[i,:] = 10*np.log10(invec)

        # print progress
        progress_fft = int((i+1)*100/(Time/Dt))
        print progress_fft
        sys.stdout.write("\033[F")

   f.close()
   #print(i+1)

   # Frequency of the selected bandwidth
   freq = bandwidth[lfreq:rfreq+1]

   return I, freq, time
