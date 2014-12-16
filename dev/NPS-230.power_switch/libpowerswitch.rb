#!/usr/bin/env ruby

require 'net/telnet'
require 'rubygems'
require 'highline/import'

pwd = ask("Enter password: ") { |q| q.echo = false }

#retrieve IP of power switch
ip = File.open("#{File.expand_path(File.dirname(__FILE__))}/ip.txt","r").read


powerswitch = Net::Telnet::new("Host" => ip,
                             "Timeout" => 3,
                             "Prompt" => /[$%#>] \z/n)

powerswitch.cmd(pwd) { |c| print c }
powerswitch.cmd("/X") { |c| print c }
powerswitch.close