#!/bin/bash

function antenna_name
{
	[ $# -ne 1 ] && return 3
	case $1 in
	1)
		echo VHF
	;;
	2)
		echo S-Band
	;;
	3)
		echo UHF
	;;
	*)	
		return 3
	;;
	esac
	return 0
}

INFO_MSG="'1' is for $(antenna_name 1), '2' is for $(antenna_name 2) (may not be working) and '3' is for $(antenna_name 3)"

#handle inputs
if [ $# -eq 0 ]
then
	echo "Input is empty. You need to specify antenna: $INFO_MSG"
	exit 3
else
	ANT=$1
	ANT_NAME=$(antenna_name $ANT)
	#sanity
	if [ $? -ne 0 ] 
	then
		echo "Unknown antenna '$ANT', need a valid antenna: $INFO_MSG"
		exit 3
	fi
fi

# set radio's center frequency
ANT="$(printf "%1i" $ANT)"
echo -n "Changing antenna to $ANT_NAME ($ANT)... "
echo -en "AN$ANT\r" > /dev/ttyUSB0
echo "Done!"

