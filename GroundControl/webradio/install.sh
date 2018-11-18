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
	sudo apt-get install fftw3 libjsoncpp-dev libmicrohttpd-dev libmp3lame-dev librtlsdr-dev libpulse-dev autoconf || exit $?
;;
2)
	echo "step 2: clone and update from gituhub"
	if [ ! -d $DIR/github ]
	then
		mkdir $DIR/github
 		git clone https://github.com/kpreid/shinysdr.git $DIR/github
	fi
	#update
	cd $DIR/github
	git pull
	cd - > /dev/null
;;
3)
	echo "step 3: build"
	cd $DIR/github
	autoreconf --install
	./configure
	make
	cd - > /dev/null
;;
4)
	echo "step 4: run it!"
	cd $DIR/github/src
	./webradio
	cd - > /dev/null
;;
*)
        echo "ERROR: unknown step $1"
        exit 3
;;
esac

