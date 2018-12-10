#!/usr/bin/python
#
# This function extracts data from the meta-file and starts a recording using the USRP
#
# Development log:
#
#	- Bart Root, 14-12-2015: Initial development
#
########################################################################

# import libraries

import sys, getopt
import os
import os.path
import yaml
import subprocess
import datetime

# initialisation of global variables

LOC_REC = '/media/data/'
LOC_ARM = 'REC_ARMED/'
LOC_RAD = '/home/doptrack/DopTrack/dev/AR5001D.radio/'
LOC_ERR = 'REC_ERROR/'

def main(argv):
   inputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:",["ifile="])
   except getopt.GetoptError:
      print 'test.py -i <inputfile>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'test.py -i <inputfile>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg

   # load meta-file
   inputstr = LOC_ARM + str(inputfile)
   with open(inputstr, 'r') as metaf:
        meta = yaml.load(metaf)  

   # set the parameters for the recording
   NORADID = meta['Sat']['State']['NORADID']
   name = meta['Sat']['State']['Name']
   freq = meta['Sat']['State']['Tuning Frequency']
   antenna = meta['Sat']['State']['Antenna']
   samp_rate = meta['Sat']['Record']['sample_rate']
   num_samp = meta['Sat']['Record']['num_sample']
   STime = meta['Sat']['Record']['Start of recording']

   # set radio
   BACK = os.getcwd()
   os.chdir(LOC_RAD)
   # set Antenna
   subprocess.call(['./setRadio_Antenna.sh',str(antenna)])
   # set frequency
   subprocess.call(['./setRadio_frequency.sh',str(freq)])
   # exit remote acces radio
   subprocess.call('./exit_Radio.sh')
   os.chdir(BACK)
   
   # set recording
   filename = str(name) + '_' + str(NORADID) + '_' + str(STime)
   start_rec_cmd = 'uhd_rx_cfile -a "addr=192.168.10.3" -f 45.05M --samp-rate=' + str(samp_rate) + ' -N ' + str(num_samp) + ' ' + str(LOC_REC) + filename + '.32fc'
   time3 = datetime.datetime.now()
   # get time1
   time1 = datetime.datetime.utcnow()

   # start recording
   #print start_rec_cmd
   #subprocess.call(start_rec_cmd)
   os.system(start_rec_cmd)

   # get time 2
   time2 = datetime.datetime.utcnow()

   # fill in the rest of meta file
   meta['Sat']['Record']['time1 UTC'] = time1
   meta['Sat']['Record']['time2 UTC'] = time2
   meta['Sat']['Record']['time3 LT'] = time3

   # put data meta file in recording directory
   data_out = LOC_REC + filename + '.32fc'
   if os.path.isfile(data_out):
       meta_out = LOC_REC + filename + '.yml'              
       with open(meta_out, 'w') as outfile:
          outfile.write( yaml.dump(meta, default_flow_style=False) )
       outfile.close()
       os.remove(inputstr)

       # add DRRE script if Recording is Delfi-C3
#       if str(NORADID) == '32789':
 #          # run DRRE file 
  #         sys.path.append('../DRRE/')  
   #        import drre
    #        
     #      dr = drre.DRRE(filename, LOC_REC)
      #     dr.mainrun() 

   else:
       meta_err = LOC_ERR + filename + '.yml'
       with open(meta_err, 'w') as outfile:
          outfile.write( yaml.dump(meta, default_flow_style=False) )
       outfile.close()
       os.remove(inputstr)      

if __name__ == "__main__":
    main(sys.argv[1:])
