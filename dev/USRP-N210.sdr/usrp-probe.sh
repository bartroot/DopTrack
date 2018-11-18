#!/bin/bash

DIR=$( cd $(dirname $BASH_SOURCE); pwd )
IP_FILE=$DIR/sdr-ip.txt
IP=`cat $IP_FILE`

uhd_usrp_probe --args addr=$IP
