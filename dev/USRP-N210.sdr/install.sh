#!/bin/bash

#first remove usual installation of gnuradio
sudo apt-get remove gnuradio
sudo apt-get autoremove

#http://ettus-apps.sourcerepo.com/redmine/ettus/projects/uhd/wiki/GNURadio_Linux
sudo bash -c 'echo "deb http://files.ettus.com/binaries/uhd_stable/repo/uhd/ubuntu/`lsb_release -cs` `lsb_release -cs` main" > /etc/apt/sources.list.d/ettus.list'
sudo bash -c 'echo "deb http://files.ettus.com/binaries/uhd_stable/repo/gnuradio/ubuntu/`lsb_release -cs` `lsb_release -cs` main" >> /etc/apt/sources.list.d/ettus.list'
sudo apt-get update
sudo apt-get install --yes --allow-unauthenticated -t `lsb_release -cs` uhd gnuradio
sudo apt-get install --yes python python-wxgtk2.8 pyqt4-dev-tools python-qwt5-qt4 python-numpy libboost-all-dev libusb-1.0.0-dev
