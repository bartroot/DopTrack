#!/usr/bin/python
#
# This python program reads the rec.list and set ups the meta-files and arms them for recording
#
# Development log: 
#			- 14-12-2015, Bart Root: initial development
#
#-----------------------------------------------------------------------------------------------

import yaml
import Record
#import predict
import os
import subprocess
import datetime
import createYAMLfile

# set global

HOME = '/home/doptrack/'
LOC_PEN = 'REC_PENDING/'
LOC_ARM = 'REC_ARMED/'
LOC_REC = '/media/data/'

priority = 0

# make mother metafile
createYAMLfile.make()

# read rec list and loop over all satellite entries
rec_list = HOME + 'rec.list'
with open(rec_list) as f:
   for line in f:
      # priority is in sequency of read first
      priority = priority + 1
      line = line.strip()
      columns = line.split()
      
      # set satellite values for later
      NORADID = columns[1]
      name = columns[0]

      # create mother meta file for this particular satellite
      with open('empty.yml', 'r') as metaf:
           metam = yaml.load(metaf)
      metaf.close()
      
      # fill in mothe meta file
      metam['Sat']['State']['Name'] = columns[0]
      metam['Sat']['State']['NORADID'] = columns[1]
      metam['Sat']['State']['Tuning Frequency'] = columns[2]
      metam['Sat']['Record']['sample_rate'] = columns[3]
      metam['Sat']['State']['Priority'] = priority

      # Determine Antenna

      if

      # read TLE
      subprocess.call(['./getTLE',str(NORADID)])

      # make prediction
      # predict.predict()     

      # make the metafiles
      tnow = datetime.datetime.now()
      year = tnow.year
      
      # input file
      fname = 'prediction_' + str(NORADID) '.txt'
      fin = open(fname,'r')

      # Start reading the prediction file and construct the metafile
      for fline in fin.readlines():
          
          elevation = int(fline[25:27])

          # check if elevation is above a certain treshold
          if elevation > 5 :
               # get the acquired variables from the line
                day = int(fline[3:5])
                month = int(fline[6:8])
                bhour = int(fline[9:11])
                bminute = int(fline[12:14])
          
                ehour = int(fline[9:11])
                eminute = int(fline[12:14])
      
                SAzimuth = int(fline[])
                EAzimuth = int(fline[])        

                # do the calculations
                if bminute == 0 :
                        bminute = 59
                        bhour = bhour - 1
                else :
                        bminute = bminute - 1
                if eminute == 59 :
                        eminute = 0
                        ehour = ehour + 1
                else :
                        eminute = eminute + 1

                # determine the length of recording
                tb = datetime.datetime(year,month,day,bhour,bmin)
                te = datetime.datetime(year,month,day,ehour,emin)
                lofp = te.seconds - tb.seconds

                #  make daughter metafile
                meta = metam 
                # fill in the meta fill
                meta['Sat']['Record']['EAzimuth'] = 
                

           else :
                # do nothing
                elevation = elevation


print "End of for-loop!"
fin.close()
fout.close()


f.close()
# close rec list

# check priority

# ARM recordings

# Update pending list

# end of program


