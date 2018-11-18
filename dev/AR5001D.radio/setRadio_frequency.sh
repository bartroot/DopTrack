#!/bin/bash

DIR=$(cd $(dirname $0);pwd)

#handle inputs
if [ $# -eq 0 ]
then
	echo "Input is empty. You need to specify the frequency in Hz"
	exit 3
else
	freq=$1
fi

# set radio's center frequency
foo="$(printf "%010d" $freq)"
echo -en "CF${foo}\r" > /dev/ttyUSB0
echo "Radio reports tunned frequency to be $($DIR/command-ar5001d.rb CF)"
