#!/bin/bash

DIR=$( cd $(dirname $BASH_SOURCE); pwd)

FILE=$DIR/sdr-name.txt
NAME=`cat $FILE`

FILE=$DIR/sdr-ip.txt
IP=`cat $FILE`

usrp_burn_mb_eeprom --values="subnet=255.255.255.0"
usrp_burn_mb_eeprom --values="gateway=192.168.10.10"
usrp_burn_mb_eeprom --values="name=$NAME"
