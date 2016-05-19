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
import sys
import datetime
import math
from sgp4.coordconv3d import *
from sgp4.geo import WGS84
from sgp4.sidereal import *
from sgp4.earth_gravity import wgs84
from sgp4.io import twoline2rv
import time as gmttime
import json

# Set global variables

pi = 3.1415926535897
Re = 6378136.00 # radius of Earth in meters
min_part = 1  # interval between SGP4 propagation
fullPeriod = 5*60*60/min_part # 5 day period of prediction

# Get starttime and start date
TT = datetime.datetime.utcnow()
gmtoff = int(-gmttime.timezone)/3600

# station coordinates [station: DopTrack]
station_lat = 51+59/60+56.376/60/60
station_lon = 4+22/60+24.551/60/60
station_h = 0#130.85

# Initializing
#print "Start of reading TLE file..."

# -------------------input TLE------------------------------------
fname = 'TLE.txt'
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
print "Start 5-min prediction at: %s" % (TT)

# open file to write orbit values
#f2 = open('Prediction_5day.txt','w')

# scenario variables
Pass = 0
pass_hor_minus1 = 0
inview = 0
scenario = 0
k = 1
start = 0
f3 = open('website_data.txt','w')
lst = []

# start of for loop of 5-day prediction
for it in range(1,fullPeriod):
	# add time
	time = TT + datetime.timedelta(0,min_part*(it-1))
	
	Unixtime = time.strftime("%s")

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
        Az = aer[0]
        El = aer[1]
        Height = alt/1000;
  
	# get other variables
        NORADID = satellite.satnum
        Vel = math.sqrt(398600.4415/((Re+alt)/1000))
	
	# get the Right Ascension and Declination
        r2d = 180/pi
 	d2r = pi/180
  	ThetaLST = gst + lon
        Dec = math.asin( math.sin(El*d2r)*math.sin(lat*d2r) + math.cos(El*d2r) * math.cos(lat*d2r) * math.cos(Az*d2r))*r2d
	LHA = math.atan2(-math.sin(Az*d2r)*math.cos(El*d2r)/math.cos(Dec*d2r) , (math.sin(El*d2r) - math.sin(Dec*d2r) * math.sin(lat*d2r)) / (math.cos(Dec*d2r) * math.cos(lat*d2r)))*r2d
	Ra = divmod(ThetaLST-LHA,360)
	Ra = Ra[1]
	# test output
#        print lat,lon,Az,El,Ra,Dec,Height,Vel,NORADID,Unixtime
	
	# try json format
#        data = [float(lat),float(lon),float(Az),float(El),float(Ra),float(Dec),float(Height),float(Vel),int(NORADID),int(Unixtime)]
#    	json.dump(data, f3)
	
	f3.write("%6.2f|%6.2f|%8.4f|%8.4f|%8.4f|%8.4f|%6.2f|%8.5f|%5i|%10i||||||\n" % (float(lat),float(lon),float(Az),float(El),float(Ra),float(Dec),float(Height),float(Vel),int(NORADID),int(Unixtime)))

	# write to file
#	f2.write("%4i %02i %02i %02i %02i %02i %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e\n"  % (int(year),int(month),int(day),int(hour),int(minute),int(sec),pos[0],pos[1],pos[2],vel[0],vel[1],vel[2]))

#f2.close()
f3.close()

print "End of Program!"
