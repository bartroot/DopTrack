#!/bin/bash

DIR=$(cd $(dirname $0);pwd)

for i in `seq 1 8`
do
  $DIR/pwsw.sh getportn+ch=$i+portn
done
