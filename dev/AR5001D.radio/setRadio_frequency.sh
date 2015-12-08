#!/bin/bash
#

freq=$1

if [ -z "$freq" ]
then
  echo "Input is empty. You need to specify frequency in Hz!"
else
  # set radio's center frequency
  foo="$(printf "%010d" $freq)"
  echo "CF${foo}"
  echo -en "CF${foo}\r" > /dev/ttyUSB0
fi
