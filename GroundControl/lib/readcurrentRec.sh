#!/bin/bash

# This script reads the current planned recordings

sudo clear
DIR_REClist='/home/doptrack/'
DIR_ATlist='/home/doptrack/www/'

cat $DIR_REClist'rec.list' > rlist
cat $DIR_ATlist'atq_list' | awk '{print $1}'> alist

# show current satellites in planning
echo ---------------------------------------
echo Current Setup Satellite Recording List:
cat rlist 
echo ---------------------------------------

# show current planned passes

numl=$(cat alist | wc -l)

for i in `seq 1 $numl`;
do
        line=$(tail -n+$i alist | head -n1)
        sudo at -c $line | grep python | awk '{print $4}'

done

rm alist rlist

