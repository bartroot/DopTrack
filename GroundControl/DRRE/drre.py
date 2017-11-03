#/usr/bin/python3
""" This python program reads raw .32fc data + .yaml files of a frequency recording of the 
 Doptrack station. It converts the raw data into Doppler 'range-rate' data, extracting 
 the frequncy signal of the radio transmission.

The code is a python version of the initial prototype code in MATLAB, designed by TUDelft students, 
under supervision by Bart Root.
  author: Hielke Krijnen"""

import numpy as np
from math import *
import sys, os
import re
import yaml
import create_mask as cm
import os.path
import image2tf
from fourier import FourierData

class DRRE(object):
  """ Main class containing the logic for the DRRE program.
      Check __init__ for inputs to the class. 
    
  """
  # These default values can be changed, but usually are fine for the intended use
  est_signal_width = 7000
  time_window = .5
  parentfolder = os.path.dirname(__file__)
  c = 299792458                      # speed of light in m/s
  home = os.path.expanduser("~")     # cross-platform home folder of the system
  buffer_size = 2**22


  def __init__(self, filename = 'Delfi-C3_32789_201707271140', foldername = 'RRes', 
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
    self.figure_flag = True
    self.estimated_signal_frequency = est_signal_freq
    self.estimated_signal_width = est_signal_width
    self.readyml()


  def mainrun(self):
    """All functions are called here."""
    #self.I, self.freq, self.time = f3.FourierAnalysis(self.filename+'.32fc', self.time_window, 
    #  self.recording_length, self.sampling_rate, self.radio_local_frequency, self.estimated_signal_frequency, 
    #  self.estimated_signal_width, self.buffer_size)
    self.runFourier()
    #mask = cm.create_mask(self.FD.I)
    mask2 = cm.create_mask_v1(self.FD.I, self.time_window)
    im = image2tf.image2tf(self.FD, mask2, self.time_window, self.figure_flag)
    im.range_rate = self.c*(im.freq/im.fc-1)
    self.write_results(im)
    print('End of program!', "\n")


  def readyml(self):
    """ Reads 'filename'.yml to get the TLE's """
    try:
      with open(os.path.join(self.parentfolder, self.foldername, self.filename+'.yml'),'r') as f:
        yml = yaml.load(f)
        TLE1 = yml['Sat']['Predict']['used TLE line1']
        TLE2 = yml['Sat']['Predict']['used TLE line2']       
        self.TLE = [TLE1, TLE2]
        self.radio_local_frequency = yml['Sat']['State']['Tuning Frequency']  # in Hz
        self.sampling_rate         = yml['Sat']['Record']['sample_rate']      # in Hz
        self.recording_length      = yml['Sat']['Predict']['Length of pass']    # in seconds
    except:
      logging.warning('YML FILE NOT FOUND: Default values used')


  def runFourier(self):
    #file = os.path.join(self.parentfolder, self.foldername, self.filename+ '.npz')
    print("running fourier")
    self.FD = FourierData(self)
    #np.savez(file, self.I, self.freq, self.time)


  def write_results(self, im):
    with open(self.filename+'.rre2', 'w') as f:
      f.write('TCA = '+ str(im.TCA)+ '\n')
      f.write('Carrier frequency = '+ str(im.fc) + '\n')
    with open(self.filename+'.rre2', 'ab') as f:
      np.savetxt(f, np.vstack((im.t, im.freq, im.range_rate)).T, fmt=['%03.0f ', '%10.5f', '%10.6f'], 
        newline='\n', header='time frequency acc')


def list_of_files(foldername):
  from os import listdir
  path = os.path.dirname(__file__)
  foldername = os.path.join(path, foldername)
  def is_32fc(a):
    try: return a.split('.')[-1] =='32fc'
    except: return False
  return [f.split('.')[0] for f in listdir(foldername) if is_32fc(f)]


 
if __name__ == "__main__":
#  dr = DRRE()
#  dr.mainrun()
  print(list_of_files('RRes'))
  for file in list_of_files('RRes'):
    try:
      print(file)
      dr = DRRE(file)
      dr.figure_flag= 0
      dr.mainrun()
    except KeyboardInterrupt:
      break
    except: 
      print("some error occurred")
      pass
