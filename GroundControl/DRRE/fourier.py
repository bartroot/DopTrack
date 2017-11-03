import numpy as np
from math import *
import sys, os
import re
from scipy.fftpack import fft



class FourierData(object):
  NFFT = 65536
  
  def __init__(FD, DRRE):
    FD.FourierAnalysis(DRRE)

  def FourierAnalysis(FD,DRRE):
    # input filename
    file = os.path.join(DRRE.parentfolder, DRRE.foldername, DRRE.filename + '.32fc')

    # initial values
    Time = DRRE.recording_length
    Fs = DRRE.sampling_rate
    Dt = DRRE.time_window

    # Construct other information
    L = Time*Fs
    setWav = int(2*L/(Time/Dt))

    # get values for axis waterfall plot
    FD.f0 = float(DRRE.radio_local_frequency)
    FD.nx = float(Fs)/float(FD.NFFT)
    FD.ny = float(Dt)
    FD.bandwidth = np.arange(FD.f0-Fs/2+FD.nx/2,FD.f0+Fs/2-FD.nx/2+1,FD.nx)
    FD.time = np.arange(1,Time+1,FD.ny)

    # set certain zoom area
    FD.lbound = float(DRRE.SatData.estimated_signal_freq) - float(DRRE.SatData.estimated_signal_width)
    FD.rbound = float(DRRE.SatData.estimated_signal_freq) + float(DRRE.SatData.estimated_signal_width)
    # Initialize Matrices    
    LF = np.array((FD.bandwidth>FD.lbound).nonzero())
    lfreq = LF.min()+1
    RF = np.array((FD.bandwidth>FD.rbound).nonzero())
    rfreq = RF.min()+1
    numC = (FD.bandwidth[lfreq:rfreq+1]).size
    FD.I = np.zeros([int(Time/Dt),int(numC)],dtype='float')
    FD.forend = int(Time/Dt)

    with open(file,'rb') as f:
      for i in range(0,FD.forend):
        # start reading the file
        t = np.fromfile(f,dtype = np.float32, count = int(setWav))
        n = int(len(t)/2)

        # Reconstruct the complex array
        v = np.zeros(n, dtype = np.complex)
        v.real = t[::2]
        v.imag = -t[1::2]

        # generate the fft of the signal
        Y = fft(v,FD.NFFT)
        dum = abs(Y[0:FD.NFFT+1])
        half = int(len(dum)/2)

        # fill in the Waterfall matrix
        line = np.zeros(dum.shape)
        line[0:half] = dum[half:]
        line[half:] = dum[0:half]
        line = line.conj().transpose()

        invec = np.asarray(line[lfreq:rfreq+1], float)
        invec[invec<=0] = 10**(-10)         # remove divide by zeros, possibly superfluous
        #FD.I[i,:] = 10*np.log10(invec)
        FD.I[i,:] = invec

    # Frequency of the selected bandwidth
    FD.freq = FD.bandwidth[lfreq:rfreq+1]
