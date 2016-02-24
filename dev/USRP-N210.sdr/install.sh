#!/bin/bash

if [ $# -eq 1 ]
then
  METHOD=$1
else
  echo -e "ERROR: Need one of the following methods:\n$( grep method-anchor $0 | tail -n +2 | sed 's/) #method-anchor/ /g' )"
  exit 3
fi

case $METHOD in
ettus) #method-anchor
  #http://ettus-apps.sourcerepo.com/redmine/ettus/projects/uhd/wiki/GNURadio_Linux
  sudo bash -c 'echo "deb http://files.ettus.com/binaries/uhd_stable/repo/uhd/ubuntu/`lsb_release -cs` `lsb_release -cs` main" > /etc/apt/sources.list.d/ettus.list'
  sudo bash -c 'echo "deb http://files.ettus.com/binaries/uhd_stable/repo/gnuradio/ubuntu/`lsb_release -cs` `lsb_release -cs` main" >> /etc/apt/sources.list.d/ettus.list'
  sudo apt-get update
  sudo apt-get install --yes --allow-unauthenticated -t `lsb_release -cs` uhd gnuradio
  sudo apt-get install --yes python python-wxgtk2.8 pyqt4-dev-tools python-qwt5-qt4 python-numpy libboost-all-dev libusb-1.0.0-dev
;;
ettus-rm) #method-anchor
  sudo apt-get purge --auto-remove gnuradio uhd python-wxgtk2.8 pyqt4-dev-tools python-qwt5-qt4 python-numpy libboost-all-dev libusb-1.0.0-dev
  sudo rm -fv /etc/apt/sources.list.d/ettus.list
  sudo apt-get update
;;
ubuntu) #method-anchor
  #fauhdlc - experimental VHDL compiler and interpreter
  #gr-osmosdr - Gnuradio blocks from the OsmoSDR project
  #libfauhdli-dev - interpreter library and development files for fauhdli
  #libgnuradio-uhd3.7.2.1 - gnuradio universal hardware driver functions
  #libuhd-dev - universal hardware driver for Ettus Research products
  #libuhd003 - universal hardware driver for Ettus Research products
  #uhd-host - universal hardware driver for Ettus Research products
  sudo apt-get install --yes gnuradio uhd-host
;;
ubuntu-rm) #method-anchor
  sudo apt-get purge --auto-remove gnuradio uhd-host libgnuradio* libuhd*
;;
myriad) #method-anchor
  for i in ppa:myriadrf/drivers ppa:myriadrf/gnuradio
  do
    sudo add-apt-repository -y $i
  done
  sudo apt-get update
  sudo apt-get install --yes gnuradio uhd
;;
myriad-rm) #method-anchor
  sudo apt-get purge --auto-remove gnuradio uhd libgnuradio* libuhd* libvolk1-bin
  for i in ppa:myriadrf/drivers ppa:myriadrf/gnuradio
  do
    sudo apt-add-repository -y --remove $i
  done
  sudo apt-get update
;;
gqrx) #method-anchor
  #http://gqrx.dk/download/install-ubuntu
  for i in ppa:bladerf/bladerf ppa:ettusresearch/uhd ppa:myriadrf/drivers ppa:myriadrf/gnuradio ppa:gqrx/gqrx-sdr
  do
    sudo add-apt-repository -y $i
  done
  sudo apt-get update
  sudo apt-get install --yes gqrx-sdr libvolk1-bin
  volk_profile
;;
gqrx-rm) #method-anchor
  sudo apt-get purge --auto-remove gqrx* libvolk1-bin
  for i in ppa:bladerf/bladerf ppa:ettusresearch/uhd ppa:myriadrf/drivers ppa:myriadrf/gnuradio ppa:gqrx/gqrx-sdr
  do
    sudo apt-add-repository -y --remove $i
  done
  sudo apt-get update
;;
*)
  echo "ERROR: unknown install method '$METHOD'"
;;
esac

