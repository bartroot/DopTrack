#/usr/bin/python
#
#
# This python program reads raw .32fc data + .yaml files of a frequency recording of the 
# Doptrack station. It converts the raw data into Doppler 'range-rate' data, extracting 
# the frequncy signal of the radio transmission.
#
# The code is a python version of the initial prototype code in MATLAB, designed by TUDelft students:
# 	-
# 	-
# 	-
# 	-
#
# Development log: 
#                       - 24-07-2016, Bart Root: initial development
#
#-----------------------------------------------------------------------------------------------

# import other fuctions

import sys, getopt
import yaml
import FourierAnalysis as FFT
import create_mask as CM

def main(argv):
   datafile = ''
   ymlfile = ''
   try:
      opts, args = getopt.getopt(argv,"hd:y:",["dfile=","yfile="])
   except getopt.GetoptError:
      print 'DRRE.py -d <datafile> -y <ymlfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'DRRE.py -d <datafile> -y <ymlfile>'
         sys.exit()
      elif opt in ("-d", "--dfile"):
         datafile = arg
      elif opt in ("-y", "--yfile"):
	 ymlfile = arg
   
   # read yml file file
   with open(ymlfile, 'r') as yfileread:
        yml = yaml.load(yfileread) 

   # Local Recording Settings
   radio_local_frequency = yml['Sat']['State']['Tuning Frequency']  # in Hz
   sampling_rate         = yml['Sat']['Record']['sample_rate']      # in Hz
   recording_length      = yml['Sat']['Predict']['Length of pass']    # in seconds

   # Orbit & Satellite Parameters
   TLE1 = yml['Sat']['Predict']['used TLE line1'] 
   TLE2 = yml['Sat']['Predict']['used TLE line2']

   # Program Settings
   estimated_signal_frequency = 145888300 	# in Hz **TBM: Need to get the lasts DRRE freq estimate**
   time_window = 1 				# in seconds
   estimated_signal_width = 7000		# in Hz **TBM: need to be linked to height (Vc) of sat**  
   c = 299792458                      		# speed of light in m/s

   # Hardware Settings
   buffer_size = 2^22 				# in bits

   #-----------------------------------
   # END USER INPUT
   #-----------------------------------

   # Fourier Transformation of the signal and extraction of subset around the expected signal
   print 'Start Fourier Transformation of the raw data'
   I, f, t = FFT.FourierAnalysis(datafile, time_window, recording_length, sampling_rate, radio_local_frequency, estimated_signal_frequency, estimated_signal_width, buffer_size)
 
   # testing limit
   #print I
   #print 'frequency'
   #print f
   #print 'time'
   #print t
  
   # Create mask to remove vertical and horizontal lines from dataset
   print 'Construct Initial filter Mask'
   mask = CM.create_mask(I)

   # Extracting the time and frequency, central frequency, and TCA
   print 'Extract range-rate from frequency spectrum' 
   #***TBD***
   #[ tsig, fsig, acc, fc, TCA ] = image2tf(I, mask, t, f, time_window, figure_flag);

   # Compute range rate
   print 'Compute range-rate from extracted frequency data'
   #***TBD***
   #range_rate = c*(fsig/fc-1);
 
   # Write the signal subsection and other results to file
   print 'Write data to file'
   # write file to same name as .32fc, but with .rre extension
   #***TBD***


if __name__ == "__main__":
   main(sys.argv[1:])
