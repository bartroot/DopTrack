#/usr/bin/python3
""" This python program reads raw .32fc data + .yaml files of a frequency recording of the 
 Doptrack station. It converts the raw data into Doppler 'range-rate' data, extracting 
 the frequncy signal of the radio transmission.

The code is a python version of the initial prototype code in MATLAB, designed by TUDelft students, under supervision by Bart Root.
  author: Hielke Krijnen"""

import numpy as np
from math import *
import sys, os
import re
import yaml
import create_mask as cm
from scipy.fftpack import fft
import os.path
import image2tf


class DRRE(object):
  """ Main class containing the logic for the DRRE program.
      Check __init__ for inputs to the class. 
    
  """
  # These default values can be changed, but usually are fine for the intended use
  est_signal_width = 7000
  time_window = 1
  parentfolder = os.path.dirname(__file__)
  c = 299792458                      # speed of light in m/s
  home = os.path.expanduser("~")     # cross-platform home folder of the system
  buffer_size = 2^22

  def __init__(self, filename = 'Delfi-C3_32789_201607132324', foldername = '', 
    est_signal_freq = 145888300, figure_flag = False, est_signal_width = 7000):
    """Initialization of the class, allows to modify the behaviour of the program.
    inputs:
    filename: string of filename without extension. defaults to newest in folder
    foldername: location of file wrt working directory TODO check defenition. defaults to newest
    est_signal_freq: estimated frequency of the satelite signal in hz. defaults to the frequency of Delfi-C3
    figure_flag: boolean to decide whether or not to show figures. defaults to false.
    """
    self.filename = filename
    self.foldername = foldername
    self.figure_flag = figure_flag
    self.estimated_signal_frequency = est_signal_freq
    self.estimated_signal_width = est_signal_width
    #if foldername == '':
    #  self.foldername = self.getfolder()
    #if filename == '':
    #  self.filename = self.getfile()
    self.readyml()

  def mainrun(self):
    # TODO: make all these fns
    # self.estimateOrbit()
    self.runFourier()
    mask = cm.create_mask(self.I)
    image2tf.filterSatData(self.I, 1/self.sampling_rate)
    # self.image2tf() // """
    # self.write_results
    # if self.figureflag: self.plots()
    pass    

  def readyml(self):
    """ Reads 'filename'.yml to get the TLE's """
    #try:
    with open(os.path.join(self.parentfolder, self.filename+'.yml'),'r') as f:
      yml = yaml.load(f)
      TLE1 = yml['Sat']['Predict']['used TLE line1']
      TLE2 = yml['Sat']['Predict']['used TLE line2']       
      self.TLE = [TLE1, TLE2]
      self.radio_local_frequency = yml['Sat']['State']['Tuning Frequency']  # in Hz
      self.sampling_rate         = yml['Sat']['Record']['sample_rate']      # in Hz
      self.recording_length      = yml['Sat']['Predict']['Length of pass']    # in seconds
    #except:
    #  print('YML FILE NOT FOUND: Default values used')

  def getfolder(self):
    """ uses the latest modified folder containig the correct filetype
    TODO: needed? preferable behaviour?
    """
    all_subdirs = [d for d in os.listdir(self.home) if os.path.isdir(d)]
    return max(all_subdirs, key=os.path.getmtime)

  def getfile(self):
    return ''

  def runFourier(self):
    file = os.path.join(self.parentfolder, self.filename+ '.npz')
    try:
      with open(file, 'rb') as f:
        load = np.load(f)
        self.I = load['arr_0']
        self.freq = load['arr_1']
        self.time = load['arr_2']
    except:
      print("running fourier")
      self.FourierAnalysis()
      np.savez(file, self.I, self.freq, self.time)

  def FourierAnalysis(self):
    # input filename
    file = os.path.join(self.parentfolder, self.filename + '.32fc')

    # initial values
    Time = self.recording_length
    Fs = self.sampling_rate
    Dt = self.time_window

    # Construct other information
    L = Time*Fs
    setWav = int(2*L/(Time/Dt))

    NFFT = 65536

    # get values for axis waterfall plot
    f0 = float(self.radio_local_frequency)
    nx = float(Fs)/float(NFFT)
    ny = float(Dt)
    bandwidth = np.arange(f0-Fs/2+nx/2,f0+Fs/2-nx/2+1,nx)
    self.time = np.arange(1,Time+1,ny)

    # set certain zoom area
    lbound = float(self.estimated_signal_frequency) - float(self.estimated_signal_width)
    rbound = float(self.estimated_signal_frequency) + float(self.estimated_signal_width)
    # Initialize Matrices    
    LF = np.array((bandwidth>lbound).nonzero())
    lfreq = LF.min()+1
    RF = np.array((bandwidth>rbound).nonzero())
    rfreq = RF.min()+1
    numC = (bandwidth[lfreq:rfreq+1]).size
    self.I = np.zeros([int(Time/Dt),int(numC)],dtype='float')
    forend = int(Time/Dt)

    with open(file,'rb') as f:
      for i in range(0,forend):
        # start reading the file
        t = np.fromfile(f,dtype = np.float32, count = setWav)
        n = int(len(t)/2)

        # Reconstruct the complex array
        v = np.zeros(n, dtype = np.complex)
        v.real = t[::2]
        v.imag = -t[1::2]

        # generate the fft of the signal
        Y = fft(v,NFFT)
        dum = 2*abs(Y[0:NFFT+1]/setWav)
        half = int(len(dum)/2)

        # fill in the Waterfall matrix
        line = np.zeros(dum.shape)
        line[0:half] = dum[half:]
        line[half:] = dum[0:half]
        line = line.conj().transpose()

        invec = np.asarray(line[lfreq:rfreq+1], float)
        invec[invec<=0] = 10**(-10)         # To remove divide by zeros, possibly superfluous
        self.I[i,:] = 10*np.log10(invec)

    # Frequency of the selected bandwidth
    self.freq = bandwidth[lfreq:rfreq+1]



def main(argv):
  """TODO, dit is nog niet de intended use schat ik. 
  Geeft iig een idee"""
  args = {}
  for i, arg in enumerate(argv):
    for j, match in enumerate(['filename', 'foldername', 'signal_freq', 'figure_flag', 'signal_width']):
      if re.match(arg, match):
        args[j] = argv[i+1]
  args = sort(args)

  dr = DRRE([item for item, arg in args])
  dr.mainrun()


 
if __name__ == "__main__":
  dr = DRRE()
  dr.mainrun()
  #cm.create_mask(dr.I)
  #print(dr.runFourier())
  #main(sys.argv)