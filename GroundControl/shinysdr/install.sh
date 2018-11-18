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
	sudo apt-get install --yes python gnuradio || exit 3
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
	python $DIR/github/setup.py build
;;
4) 	
        echo "step 4: run the script fetch-js-deps.sh"
        cd $DIR/github
        ./fetch-js-deps.sh
        cd - > /dev/null
;;
5)
        get_sudo || exit $?
        echo "step 5: install"
	cd $DIR/github
        sudo python setup.py install
	cd - > /dev/null
;;
6)
	echo "step 6: create config"
	shinysdr --create doptrack.cfg
;;
*)
	echo "ERROR: unknown step $1"
	exit 3
esac

