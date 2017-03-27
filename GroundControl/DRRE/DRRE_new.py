#/usr/bin/python3
""" This python program reads raw .32fc data + .yaml files of a frequency recording of the 
 Doptrack station. It converts the raw data into Doppler 'range-rate' data, extracting 
 the frequncy signal of the radio transmission.

The code is a python version of the initial prototype code in MATLAB, designed by TUDelft students:
  author: Hielke Krijnen"""

import numpy as np
from math import *
import sys, os
import re

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




  def __init__(self, filename = '', foldername = '', est_signal_freq = 145888300, figure_flag = False, est_signal_width = 7000):
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
    self.est_signal_freq = est_signal_freq
    if foldername == '':
      self.foldername = self.getfolder()
    if filename == '':
      self.filename = self.getfile()
    self.TLE = self.getTLE()

  def mainrun(self):
    # TODO: make all these fns
    # self.estimateOrbit()
    # self.extractSignal() // possibly class?
    # self.image2tf() // """
    # self.write_results
    # if self.figureflag: self.plots()

    

  def getTLE(self):
    """ Reads 'filename'.yml to get the TLE's
    TODO is string best data type?
    """
    try:
      with open(os.path.join(self.home, self.filename+'.yml'),'r') as f:
        TLE1 = f.readline
        TLE2 = f.readline
        return [TLE1, TLE2]
    except:
      return '1 32789U 08021G   15306.88561006  .00005080  00000-0  40073-3 0  9999 2 32789 097.6403 005.0192 0008403 301.8696 110.8549 15.01151259407778 '

  def getfolder(self):
    """ uses the latest modified folder containig the correct filetype
    TODO: needed? preferable behaviour?
    """
    all_subdirs = [d for d in os.listdir(self.home) if os.path.isdir(d)]
    return max(all_subdirs, key=os.path.getmtime)

  def getfile(self):
    return ''


def main(argv):
  """TODO, dit is nog niet de intended use schat ik. 
  Geeft iig een idee"""

  args = {}
  for i, arg in enumerate(argv):
    for j, match in enumerate(['filename', 'foldername', 'signal_freq', 'figure_flag', 'signal_width']):
      if re.match(arg, match):
        args{j} = argv[i+1]
  args = sort(args)

  dr = DRRE([item for item, arg in args])
  dr.mainrun()


 
if __name__ == "__main__":
  main(sys.argv)