#!/usr/bin/env ruby

require "serialport"

#opening port
sp = SerialPort.new("/dev/ttyUSB0",115200,8,1,SerialPort::NONE)

#sending command
sp.write(ARGV[0]+"\r")

#reading feedback
out=String.new
until out[-1] == "\n"
	out+=sp.read(1)
end
puts out if out.length>0
