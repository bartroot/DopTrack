#!/bin/bash -u

DIR=$( cd $( dirname $0);pwd)

function get_sudo
{
if [[ $EUID -ne 0 ]]; then
        echo "ERROR: need sudo"
        return 3
else
        return 0
fi
}

case $1 in
1)
	get_sudo || exit $?
	echo "step 1: retrieve dependencies"
	sudo apt-get install rtl-sdr nmap libfftw3-dev libusb-1.0-0-dev || exit $?
;;
2)
	echo "step 2: retrieve libcsdr"
	if [ ! -d $DIR/libcsdr ]
        then
		mkdir $DIR/libcsdr
		git clone https://github.com/simonyiszk/csdr.git $DIR/libcsdr
	fi
        #update
        cd $DIR/libcsdr
        git pull
	make
        cd - > /dev/null
;;
3)
	get_sudo || exit $?
	echo "step 3: install libcsdr"
	cd $DIR/libcsdr
	sudo make install
	cd - > /dev/null
;;
4)
	echo "step 4: clone and update from gituhub"
	if [ ! -d $DIR/github ]
	then
		mkdir $DIR/github
 		git clone https://github.com/simonyiszk/openwebrx.git $DIR/github
	fi
	#update
	cd $DIR/github
	git pull
	cd - > /dev/null
;;
5)
	echo "step 5: build"
	cd $DIR/github
	autoreconf --install
	./configure
	make
	cd - > /dev/null
;;
6)
	echo "step 6: configure it!"
	vi $DIR/github/config_webrx.py
;;
7)
	echo "step 7: run it"
	cd $DIR/github
	./openwebrx.py
	cd - > /dev/null
;;
*)
        echo "ERROR: unknown step $1"
        exit 3
;;
esac

