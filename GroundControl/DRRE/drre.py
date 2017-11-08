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
import satdata_delfi as delfi


class DRRE(object):
  """ Main class containing the logic for the DRRE program.
      Check __init__ for inputs to the class. 
    
  """
  # These default values can be changed, but usually are fine for the intended use
  time_window = .5
  parentfolder = os.path.dirname(__file__)
  c = 299792458                      # speed of light in m/s
  #buffer_size = 2**22

  def __init__(self, filename = 'Delfi-C3_32789_201707271140', foldername = 'RRes', destination = 'same', figure_flag = False, SatData=delfi.SatData()):
    """Initialization of the class, allows to modify the behaviour of the program.
    inputs:
    filename: string of filename without extension. defaults to newest in folder
    foldername: path of file wrt working directory 
    est_signal_freq: estimated frequency of the satelite signal in hz. defaults to the frequency of Delfi-C3
    figure_flag: boolean to decide whether or not to show figures. defaults to false.
    destination: allows rre file to be saved to a different location, defaults to the same as foldername
    """
    self.filename = filename
    self.foldername = foldername
    self.figure_flag = figure_flag
    self.SatData = SatData
    self.destination = destination
    if destination == 'same':
      self.destination = foldername    
    self.readyml()


  def mainrun(self):
    """All functions are called here."""
    self.FD = FourierData(self)
    mask2 = cm.create_mask_v1(self.FD.I, self.time_window)
    self.im = image2tf.image2tf(self.FD, self.SatData, mask2, self.time_window, self.figure_flag)
    self.im.range_rate = self.c*(self.im.freq/self.im.fc-1)
    self.write_results()
    print('End of program!', "\n")

  def runFourier(self):
    self.FD = FourierData(self)

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
      print('YML FILE NOT FOUND: Default values used')


  def write_results(self):
    with open(os.path.join(self.parentfolder, self.destination, self.filename+'.rre2'), 'w') as f:
      f.write('TCA = '+ str(self.im.TCA)+ '\n')
      f.write('Carrier frequency = '+ str(self.im.fc) + '\n')
    with open(os.path.join(self.parentfolder, self.destination, self.filename+'.rre2'), 'ab') as f:
      np.savetxt(f, np.vstack((self.im.t, self.im.freq, self.im.range_rate)).T, fmt=['%03.0f ', '%10.5f', '%10.6f'], 
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
  dr = DRRE('Delfi-C3_32789_201709301034', figure_flag=True)
  dr.mainrun()
  """
  for file in list_of_files('RRes'):
    try:
      print(file)
      dr = DRRE(file)
      dr.mainrun()
    except KeyboardInterrupt:
      break
    except: 
      print("some error occurred")
      pass
"""