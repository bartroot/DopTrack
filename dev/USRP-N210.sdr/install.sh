#!/bin/bash

METHOD=ubuntu

#first remove current installation of gnuradio
sudo apt-get remove gnuradio uhd python-wxgtk2.8 pyqt4-dev-tools python-qwt5-qt4 python-numpy libboost-all-dev libusb-1.0.0-dev
sudo apt-get autoremove

case $METHOD in
ettus)
  #http://ettus-apps.sourcerepo.com/redmine/ettus/projects/uhd/wiki/GNURadio_Linux
  sudo bash -c 'echo "deb http://files.ettus.com/binaries/uhd_stable/repo/uhd/ubuntu/`lsb_release -cs` `lsb_release -cs` main" > /etc/apt/sources.list.d/ettus.list'
  sudo bash -c 'echo "deb http://files.ettus.com/binaries/uhd_stable/repo/gnuradio/ubuntu/`lsb_release -cs` `lsb_release -cs` main" >> /etc/apt/sources.list.d/ettus.list'
  sudo apt-get update
  sudo apt-get install --yes --allow-unauthenticated -t `lsb_release -cs` uhd gnuradio
  sudo apt-get install --yes python python-wxgtk2.8 pyqt4-dev-tools python-qwt5-qt4 python-numpy libboost-all-dev libusb-1.0.0-dev
;;
ubuntu)
#fauhdlc - experimental VHDL compiler and interpreter
#gr-osmosdr - Gnuradio blocks from the OsmoSDR project
#libfauhdli-dev - interpreter library and development files for fauhdli
#libgnuradio-uhd3.7.2.1 - gnuradio universal hardware driver functions
#libuhd-dev - universal hardware driver for Ettus Research products
#libuhd003 - universal hardware driver for Ettus Research products
#uhd-host - universal hardware driver for Ettus Research products

  sudo rm -fv /etc/apt/sources.list.d/ettus.list
  sudo apt-get update
  sudo apt-get install --yes gnuradio uhd-host
;;
*)
  echo "ERROR: unknown install method '$METHOD'"
;;
esac

