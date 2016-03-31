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
import predict
import os
import subprocess
import datetime
import createYAMLfile

# set global

HOME = '/home/doptrack/'
LOC_PEN = 'REC_PENDING/'
LOC_ARM = 'REC_ARMED/'
LOC_REC = '/media/data/'
LOC_RUN = '/home/doptrack/DopTrack/GroundControl/Automation/'

priority = 0
os.chdir(LOC_RUN)

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
      freq = int(columns[2])
      samp_rate = int(columns[3])

      # create mother meta file for this particular satellite
      with open('empty.yml', 'r') as metaf:
           metam = yaml.load(metaf)
      metaf.close()
      
      # fill in mothe meta file
      metam['Sat']['State']['Name'] = columns[0]
      metam['Sat']['State']['NORADID'] = columns[1]
      metam['Sat']['State']['Tuning Frequency'] = int(columns[2])
      metam['Sat']['Record']['sample_rate'] = int(columns[3])
      metam['Sat']['State']['Priority'] = priority

      # Determine Antenna
      if 30000000 < freq <= 300000000:
         # VHF antenna range
         metam['Sat']['State']['Antenna'] = 1 
      elif 300000000 < freq < 1000000000:
         #UHF antenna range
         metam['Sat']['State']['Antenna'] = 3
      else:
         # default: VHF antenna range
         metam['Sat']['State']['Antenna'] = 1

      # read TLE
      subprocess.call(['./getTLE',str(NORADID)])

      # make prediction
      metam = predict.predict(metam)     

      # make the metafiles
      tnow = datetime.datetime.now()
      year = tnow.year
      
      # input file
      fname = 'prediction_' + str(NORADID) + '.txt'
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
          
                ehour = int(fline[30:32])
                eminute = int(fline[33:35])
      
                SAzimuth = int(fline[15:18])
                EAzimuth = int(fline[36:39])        

                # do the calculations
                if bminute == 0 :
                        bminute = 59
			if bhour == 0 :
				bhour = 23
			else :
                        	bhour = bhour - 1
                else :
                        bminute = bminute - 1
                if eminute == 59 :
                        eminute = 0
			if ehour == 23 :
				ehour = 0
			else : 
                        	ehour = ehour + 1
                else :
                        eminute = eminute + 1

                # determine the length of recording
                tb = datetime.datetime(year,month,day,bhour,bminute)
                te = datetime.datetime(year,month,day,ehour,eminute)
                lofp = te - tb
                lofp = lofp.seconds

                # determine number of samples
                num_samp = int(lofp)*int(samp_rate)

                # start of recording
                start_rec = int(str(year) + str(month).zfill(2) + str(day).zfill(2) + str(bhour).zfill(2) + str(bminute).zfill(2))

                #  make daughter metafile
                meta = metam 
                # fill in the meta fill
                meta['Sat']['Predict']['EAzimuth'] = EAzimuth
                meta['Sat']['Predict']['SAzimuth'] = SAzimuth
                meta['Sat']['Predict']['Length of pass'] = int(lofp)
                meta['Sat']['Predict']['Elevation'] = elevation

                meta['Sat']['Record']['Start of recording'] = start_rec
                meta['Sat']['Record']['num_sample'] = num_samp

                # Stored the metafile in the pending direcory
                metaname = LOC_PEN + str(name) + '_' + str(NORADID) + '_' + str(start_rec) + '.yml'
                with open(metaname, 'wr') as outfile:
                   outfile.write( yaml.dump(meta, default_flow_style=False) )
                outfile.close()
          else:
                # do nothing
                elevation = elevation


print "End of for-loop!"
fin.close()
# close rec list
f.close()

# check the pending recordings for arm
subprocess.call(['./make_atq.sh'])

# Log list armed recordings

# end of program


