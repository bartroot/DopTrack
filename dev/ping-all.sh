#!/bin/bash -u

for i in `find . -maxdepth 1 -type d`
do
  [ ! -e $i/ip.txt ] && continue
  IP=`cat $i/ip.txt`
  ping $IP -W 1 -A -c 5 > /dev/null && echo $i is connected || echo $i is NOT connected!
done
