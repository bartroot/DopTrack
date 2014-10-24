#!/bin/bash

DIR=$( cd $(dirname $BASH_SOURCE); pwd )
IP_FILE=$DIR/sdr-ip.txt
IP=`cat $IP_FILE`

if ping $IP -W 1 -A -c 5 > /dev/null
then
	echo Ping $IP succeeded.
else
	echo Ping $IP failed!
	exit 3
fi

for i in addr=$IP type=usrp2 serial=F4A0B9 name=sdr
do
	echo "trying to find USRP by $i"
	uhd_find_devices --args="$i"
done
