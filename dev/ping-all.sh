#!/bin/bash -u

DIR=$( cd "$(dirname "$(readlink -f "$BASH_SOURCE")")"; pwd)

for i in `find $DIR -maxdepth 1 -type d`
do
  [ ! -e $i/ip.txt ] && continue
  IP=`cat $i/ip.txt`
  ping $IP -W 1 -A -c 5 > /dev/null && echo $(basename $i) is connected || echo $(basename $i) is NOT connected!
done
