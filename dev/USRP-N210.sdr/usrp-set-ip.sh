#!/bin/bash

DIR=$( cd $(dirname $BASH_SOURCE); pwd )
IP_FILE=$DIR/sdr-ip.txt
IP=`cat $IP_FILE`

ETH_FILE=$DIR/sdr-eth.txt
ETH=`cat $ETH_FILE`

sudo /usr/lib/uhd/utils/usrp2_recovery.py --ifc=$ETH --new-ip=$IP
