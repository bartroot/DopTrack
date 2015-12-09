#!/usr/bin/python
#
# this program will read a TLE file and produce a 5-day prediction output file
#
# Written by Bart Root, TUDelft, 24-Aug-2015
#
# Dependent functions:
#
#	- earth_gravity.py
#	- io.py
#	-
#
# Change log:
#
# Initial developement: Bart Root - 24-Aug-2015
#
#---------------------- start of routine --------------------------------------

# import libaries
import numpy as np
import os.path
import sys, getopt
import datetime
import math
from sgp4.coordconv3d import *
from sgp4.geo import WGS84
from sgp4.sidereal import *
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import time as gmttime
import scipy.io as sio

# input arguments handling

def main(argv):
   inputfile = ''
   outputfile = ''
   start_time = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:t:",["ifile=","ofile=","tfile"])
   except getopt.GetoptError:
      print 'test.py -i <inputfile> -o <outputfile> -t <start_time> ["%d %b %Y %I:%M"]'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'test.py -i <inputfile> -o <outputfile> -t <tart_time> ["%d %b %Y %I:%M"]'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
      elif opt in ("-t", "--tfile"):
         start_time = arg
   
   # input filename
   fname = inputfile
   # Set global variables

   pi = 3.1415926535897
   Re = 6378136.00 # radius of Earth in meters
   min_part = 1   # interval between SGP4 propagation
   fullPeriod = 1*60*60/min_part # 1 hour period of prediction

   # Get starttime and start date
   #TT = datetime.datetime.utcnow()
   TT = datetime.datetime.strptime(str(start_time), '%d %m %Y %I:%M')
   
   # Initializing
   print "Start of reading TLE file..."

   # -------------------input TLE------------------------------------
   # check if TLE file is present
   if os.path.isfile(fname): 
   	f = open(fname,'r')
   else:
	sys.exit('Error: TLE.txt file is not present!')

   # continue program if TLE file is present
   line1 = f.readline()
   line2 = f.readline()
   f.close()
   # -------------------input TLE------------------------------------

   # Construct satellite variable
   satellite = twoline2rv(line1, line2, wgs84)
   print "Start propagation at: %s" % (TT)

   # open file to write orbit values
   f2 = open(outputfile,'w')

   # scenario variables
   OMEGAE = 7.29211586e-5;  # Earth rotation rate in rad/s

   # start of for loop of 5-day prediction
   for it in range(1,fullPeriod):
	# add time
	time = TT + datetime.timedelta(0,min_part*(it-1))
	
	year = time.year
	month = time.month
	day = time.day
	hour = time.hour
	minute = time.minute
	sec = time.second 

	pos,vel = satellite.propagate(year, month, day, hour, minute, sec)
	
	# Get Julian date
	jdTT = JulianDate.fromDatetime(time)
	jdut1 = JulianDate.__float__(jdTT)
	
	# Get Greenwich Apparent Siderial Time
	tut1= ( jdut1 - 2451545.0 ) / 36525.0	

	temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + (876600.0 * 3600.0 + 8640184.812866) * tut1 + 67310.54841
	
        # 360/86400 = 1/240, to deg, to rad
        
	dumy2 = divmod( temp*pi/180.0/240.0,2*pi )
	temp = dumy2[1]
        
        # ------------------------ check quadrants --------------------
        if ( temp < 0.0 ):
            temp = temp + 2*pi

        gst = temp
	cgast = math.cos(gst)
	sgast = math.sin(gst)

	# Transformation of coordinates of satellite
	x =  pos[0] * cgast * 1000 + pos[1] * sgast * 1000
	y = -pos[0] * sgast * 1000 + pos[1] * cgast * 1000
	z =  pos[2] * 1000
	vx = vel[0] * cgast * 1000 + vel[1] * sgast * 1000 + OMEGAE * y
 	vy =-vel[1] * sgast * 1000 + vel[2] * cgast * 1000 - OMEGAE * x
	vz = vel[2] * 1000

	# write to file
	f2.write("%4i %02i %02i %02i %02i %02i %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e\n"  % (int(year),int(month),int(day),int(hour),int(minute),int(sec),x,y,z,vx,vy,vz))

   f2.close()

   print "End of Program!"

if __name__ == "__main__":
	main(sys.argv[1:])
