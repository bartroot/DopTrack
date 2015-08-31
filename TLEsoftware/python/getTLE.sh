#!/bin/bash

# get TLE from Space-track.org websit

curl -c cookies.txt -b cookies.txt -k https://www.space-track.org/ajaxauth/login -d 'identity=bart_root&password=delfi-c3delfi-c3&query=https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/NORAD_CAT_ID/32789/orderby/TLE_LINE1%20ASC/format/tle' > temp_TLE.txt

if [ -s temp_TLE.txt ]
then
	rm TLE.txt
	mv temp_TLE.txt TLE.txt
else 	
	touch TLE_isempty.txt
	rm temp_TLE.txt	
fi

