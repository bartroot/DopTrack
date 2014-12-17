#!/usr/bin/env ruby

require 'net/telnet'
require 'rubygems'
require 'highline/import'

module PowerSwitch

  class Control < Struct.new(
    :connection)
    attr_reader :connection
    #retrieve IP of power switch
    @@ip=File.open("#{File.expand_path(File.dirname(__FILE__))}/ip.txt","r").read

    def initialize
      #need password
      pwd = ask("Enter password: ") { |q| q.echo = false }
      #open the connection
      @connection = Net::Telnet::new(
        "Host" => @@ip,
        "Timeout" => 3,
        "Prompt" => /[$%#>] \z/n
      )
      powerswitch.cmd(pwd) { |c| print c }
    end

    def close
      @connection.cmd("/X") { |c| print c }
      @connection.close
    end

    def cmd(comlist)
      raise RuntimeError,"Can only deal with input argument 'comlist' as an Array, not class '#{comlist.class}'." unless comlist.is_a?(Array)
      out=Array.new
      comlist.each do |c|
        out.push(@connection.cmd("/X") { |c| print c })
      end
      return out
    end

    def self.help
      PowerSwitch::Control.new.cmd("/H").close
    end

    def status
      self.cmd("")
    end

  end
end

#run status if this file is called explicitly
if __FILE__==$0
  puts PowerSwitch::Control.help
end