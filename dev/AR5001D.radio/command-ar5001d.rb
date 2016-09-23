#!/usr/bin/env ruby

require "serialport"

#set parameters
dev="/dev/ttyUSB0"
baud_rate=115200
data_bits=8
stop_bits=1
parity=SerialPort::NONE

#opening port
sp = SerialPort.new(dev,baud_rate,data_bits,stop_bits,parity)

#sending command
sp.write(ARGV[0]+"\r")

#showing response
begin
  puts sp.readline("\r")
rescue
  #do nothing
end

#close serial port
sp.close 
