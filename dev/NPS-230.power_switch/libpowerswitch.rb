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
    @@comlist={
      :help    => "H",
      :status  => "S",
      :general => "G\n\e",
      :network => "N\n\e",
      :exit    => "X",
    }
    @@power_modes=["On","Off","Boot"]

    def initialize
      #need password
      pwd = ask("Enter password: ") { |q| q.echo = false }
      #open the connection
      @connection = Net::Telnet::new(
        "Host" => @@ip,
        "Timeout" => 3,
        "Prompt" => /[$%#>] \z/n
      )
      @connection.cmd(pwd)
    end

    def close
      self.cmd(:exit)
      @connection.close
    end

    def cmd(com)
      case com
      when NilClass
        nil
      when Array
         com.map{ |c| self.cmd(c) }
      when Symbol
        raise RuntimeError,"Can not understand command #{com}'." unless @@comlist.has_key?(com)
        self.cmd(@@comlist[com])
      when String
          @connection.cmd("/"+com) { |c| print c }
      else
        raise RuntimeError,"Can only deal with input argument 'com' as an " +
          "Array, NilClass or Strings, not class '#{com.class}'." unless com.is_a?(Array)
      end
      return self
    end

    def wait(seconds)
      sleep(seconds)
    end

    def power(mode,plug)
      raise RuntimeError,"Can only deal with input argument 'mode' as a " +
          "String, not class '#{mode.class}'." unless mode.is_a?(String)
      raise RuntimeError,"Can only deal with input argument 'plug' as a " +
          "Integer, not class '#{plug.class}'." unless plug.is_a?(String)
      raise RuntimeError,"Mode has to be one of #{@@power_modes.join(',')}, not '#{mode}'." unless @@power_modes.include?(mode)
      self.cmd("#{mode} #{plug}")
    end

  end
end

#run status if this file is called explicitly
if __FILE__==$0
  puts PowerSwitch::Control.new.cmd(:help).cmd(:status).cmd(:general).cmd(:network).cmd(:status).close
end