#!/usr/bin/python
#
# this program will read a TLE file and produce a prediction of the satellite
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
import sys
import datetime
import math
from sgp4.coordconv3d import *
from sgp4.geo import WGS84
from sgp4.sidereal import *
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import time as gmttime
import yaml

def predict(meta):

   # load meta file variable
   name = meta['Sat']['State']['Name']
   NORADID = meta['Sat']['State']['NORADID']

   # Set global variables
   pi = 3.1415926535897
   Re = 6378136.00 # radius of Earth in meters
   min_part = 15   # interval between SGP4 propagation
   fullPeriod = 5*24*60*60/min_part # 5 day period of prediction

   # Get starttime and start date
   TT = datetime.datetime.utcnow()
   gmtoff = int(-gmttime.timezone)/3600

   # get time format correct
   TT_stamp = TT.strftime("%Y%m%d%H%M%S.%f")
   meta['Sat']['Predict']['time used UTC'] = TT_stamp
   meta['Sat']['Predict']['timezone used'] = gmtoff

   # station coordinates [station: DopTrack]
   station_lat = 51+59/60+56.376/60/60
   station_lon = 4+22/60+24.551/60/60
   station_h = 0#130.85

   # store station coordinates
   meta['Sat']['Station']['Name'] = 'DopTrack'
   meta['Sat']['Station']['Lat'] = station_lat
   meta['Sat']['Station']['Lon'] = station_lon
   meta['Sat']['Station']['Height'] = station_h

   # Initializing
   print "Start of reading TLE file..."
   
   # -------------------input TLE------------------------------------
   fname = 'TLE_' + NORADID + '.txt'
   # check if TLE file is present
   if os.path.isfile(fname): 
        f = open(fname,'r')
   else:
        sys.exit('Error: TLE.txt file is not present!')

   # continue program if TLE file is present
   line1 = f.readline()
   line2 = f.readline()
   f.close()

   # store in the meta file
   meta['Sat']['Predict']['used TLE line1'] = str(line1[:69]).rstrip()
   meta['Sat']['Predict']['used TLE line2'] = str(line2[:69]).rstrip()
   # -------------------input TLE------------------------------------

   # Construct satellite variable
   satellite = twoline2rv(line1, line2, wgs84)
   print "Start 5-day prediction at: %s" % (TT)

   # scenario variables
   Pass = 0
   pass_hor_minus1 = 0
   inview = 0
   scenario = 0
   k = 1
   start = 0
   predictfile = 'prediction_' + str(NORADID) + '.txt'
   f3 = open(predictfile,'w')
   lst = []

   # start of the prediction loop
   for it in range(1,fullPeriod):
	# add time
	time = TT + datetime.timedelta(0,15*(it-1))
	
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

	# Transformation to Latitude, Longitude, and altitude
        #Calculate lon
        lon = math.atan2(y, x)
        #Initialize the variables to calculate lat and alt
        alt = 0
        N = WGS84.a
        p = sqrt(x**2 + y**2)
        lat = 0
        previousLat = 90
        #Iterate until tolerance is reached
        while abs(lat - previousLat) >= 1e-9:
            previousLat = lat
            sinLat = z / (N * (1 - WGS84.e**2) + alt)
            lat = math.atan((z + WGS84.e**2 * N * sinLat) / p)
            N = WGS84.a / sqrt(1 - (WGS84.e * sinLat)**2)
            alt = p / math.cos(lat) - N

	lon = lon/pi*180
        lat = lat/pi*180

	# get station coordinates into ecef
        vlat = station_lat/180*pi
        vlon = station_lon/180*pi
        valt = station_h
        #Calculate length of the normal to the ellipsoid
        N = WGS84.a / sqrt(1 - (WGS84.e * math.sin(vlat))**2)
        #Calculate ecef coordinates
        vx = (N + valt) * math.cos(vlat) * math.cos(vlon)
        vy = (N + valt) * math.cos(vlat) * math.sin(vlon)
        vz = (N * (1 - WGS84.e**2) + valt) * math.sin(vlat)

	# check if satellite is in view of the station
	cosgamma = Re/ (Re + alt)

	# Calculate the satellite gamma angle
	satgamma = np.inner([vx,vy,vz],[x,y,z]) / math.sqrt(np.inner([x,y,z],[x,y,z])) / math.sqrt(np.inner([vx,vy,vz],[vx,vy,vz]))

	#check if satellite is inside view of horizon
	if satgamma>cosgamma:
		pass_hor = 1
	else:	
		pass_hor = 0
	
	# get elevation and azimuth values
	aer = geodetic2aer(lat, lon, alt, vlat/pi*180, vlon/pi*180, valt, ell=EarthEllipsoid(),deg=True)

	# Quantify the passes
	
	# check scenario for the first epoch
	if TT == time:
		if pass_hor == 1:
			Pass = 1
			inview = 1
			start = 1

			# initialize dynamic array
			lst.append(aer[1])			

			# satellite is in view
			scenario = 1-pass_hor
			# update logfile
			f3.write("%02i %02i-%02i %02i:%02i %3i | " % (int(Pass),int(day),int(month),int(hour+gmtoff),int(minute),int(aer[0])))
			k = k+1
		else:
			scenario = 0-pass_hor
	else:
		scenario = pass_hor_minus1 - pass_hor

	# Scenario: out of view = 1, in view = -1, no change = 0
	if scenario == 1:
		# out of view: pass ends
		inview = 0

		# Find elevation of TCA
        	if start == 1:
            		start = 0;
        	else:
            		# search for maximum Elevation in pass
            		max_EL = max(lst)
           		# update logfile
            		f3.write("%03i %02i | " % (int(aer[0]),int(max_EL)))
           		k = k+1;
			lst = []

		# update logfile
		f3.write("%02i:%02i %03i |\n" % (int(hour+gmtoff),int(minute),int(AZ_pre)))
		k=k+1	
	elif scenario == -1:
		# inview: pass begins
		Pass = Pass + 1
		inview = 1
		lst = []
		lst.append(aer[1])

		# update logfile
		f3.write("%02i %02i-%02i %02i:%02i %3i | " % (int(Pass),int(day),int(month),int(hour+gmtoff),int(minute),int(aer[0])))
		k = k+1
	elif scenario == 0:
		#no change
		lst.append(aer[1])
	else:
		sys.exit('Error: Undefined scenario at pass check!')

	# Needed for the following scenario: out of view 		
	AZ_pre = aer[0]
	EL_pre = aer[1]
	jdut1_pre = jdut1
	pass_hor_minus1 = pass_hor

   f3.close()
   # return the updated meta file
   print "End of Program!"
   return meta;

if __name__ == "__main__":
   main(meta)


