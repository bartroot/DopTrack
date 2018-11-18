#!/bin/bash

DIR=$(cd $(dirname $0);pwd)
IP=$(cat $DIR/ip.txt)

if [ $# -eq 0 ]
then
  CMD="getversion getsysclock getlog+num=01"
else
  CMD=$@
fi

URL="http://$IP/set.cmd?user=admin+pass=delfi-c3"

for i in $CMD
do
  wget -qO - $URL+cmd=$i
done


