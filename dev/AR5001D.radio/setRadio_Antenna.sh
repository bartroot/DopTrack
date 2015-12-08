#!/bin/bash
#

ANT=$1

if [ -z "$ANT" ]
then
  echo "Input is empty. You need to specify Antenna! 1 is for VHF and 3 is for UHF"
else
  # set radio's center frequency
  foo="$(printf "%1i" $ANT)"
  echo "AN${foo}"
  echo -en "AN${foo}\r" > /dev/ttyUSB0
fi
