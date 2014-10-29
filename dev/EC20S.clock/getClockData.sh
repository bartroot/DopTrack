#!/bin/bash
#
# this script reads the clock data and outputs it in one line, such that it can be given to a log file
# snmp interface must be working
#
#
# written by Bart Root, TUDelft for the Doptrack project
# date: 29-okt-2014
#
###################################################################################

# test if connection is present: TODO

TIME=`./getTime.sh | awk '{print $4" "$5}' | cut -c 2- | sed 's/.$//'`

#echo $TIME

LON=`./getLongitude.sh | awk '{print $4" "$5" "$6" "$7" "$8}' | cut -c 2- | sed 's/"//'`
LAT=`./getLatitude.sh | awk '{print $4" "$5" "$6" "$7" "$8}' | cut -c 2- | sed 's/"//'`
ALT=`./getAltitude.sh | awk '{print $4}' | cut -c 2-`

#echo $LON $LAT $ALT

NUM=`./getNumberSatLocked.sh | awk '{print $4}' | cut -c 2- | sed 's/"//'`

#echo $NUM

echo $TIME $LON $LAT $ALT $NUM 
