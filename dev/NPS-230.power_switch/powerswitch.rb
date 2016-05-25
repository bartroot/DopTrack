#!/usr/bin/env ruby

require 'net/telnet'
require 'rubygems'
require 'highline/import'

module PowerSwitch

  COMLIST={
      :help    => "H",
      :status  => "S",
      :general => "G\n\e",
      :network => "N\n\e",
      :exit    => "X",
    }
  POWERMODES=["On","Off","Boot"]

  class Control < Struct.new(
    :connection)
    attr_reader :connection
    #retrieve IP of power switch
    @@ip=File.open("#{File.expand_path(File.dirname(__FILE__))}/powerswitch.ip","r").read.chomp
    @@debug=ARGV.include?('debug')

    def initialize(pwd)
      if @@debug

        puts "Currently in debug mode, nothing happens in reality\nConneting to #{@@ip}"
      else
        #open the connection
        @connection = Net::Telnet::new(
          "Host" => @@ip,
          "Timeout" => 30,
          "Prompt" => /[$%#>] \z/n
        )
        @connection.cmd(pwd)
      end
    end

    def close
      self.cmd(:exit)
      @connection.close unless @@debug
    end

    def cmd(com)
      case com
      when NilClass
        nil
      when Array
         com.map{ |c| self.cmd(c) }
      when Symbol
        raise RuntimeError,"Can not understand command #{com}'." unless COMLIST.has_key?(com)
        self.cmd(COMLIST[com])
      when String
        if @@debug
          puts "Command: /"+com
        else
          @connection.cmd("/"+com) { |c| print c }
        end
      else
        raise RuntimeError,"Can only deal with input argument 'com' as an " +
          "Array, NilClass or Strings, not class '#{com.class}'." unless com.is_a?(Array)
      end
      return self
    end

    def power(mode,plug)
      raise RuntimeError,"Can only deal with input argument 'mode' as a " +
          "String, not class '#{mode.class}'." unless mode.is_a?(String)
      raise RuntimeError,"Can only deal with input argument 'plug' as a " +
          "Fixnum, not class '#{plug.class}'." unless plug.is_a?(Fixnum)
      raise RuntimeError,"Mode has to be one of #{POWERMODES.join(',')}, not '#{mode}'." unless POWERMODES.include?(mode)
      self.cmd("#{mode} #{plug}").cmd(:status)
    end

  end
end

#run status if this file is called explicitly
if __FILE__==$0
  #inits
  done=false
  #simple calls
  if PowerSwitch::COMLIST.keys.map{|k| k.to_s}.include?(ARGV[1])
    PowerSwitch::Control.new(ARGV[0]).cmd(ARGV[1].to_sym).close
    done=true
  end
  #power calls
  if PowerSwitch::POWERMODES.include?(ARGV[1])
    PowerSwitch::Control.new(ARGV[0]).power(ARGV[1],ARGV[2].to_i).close
    done=true
  end




  #sanity
  raise RuntimeError,"\n\n"+
    "#{__FILE__} password [#{PowerSwitch::COMLIST.keys.map{|k| k.to_s}.join('|')}|#{PowerSwitch::POWERMODES.join(' plug_nr|')} plug_nr]\n\n"+
    "First, state the password to access the power switch.\n"+
    "Then, use either:\n"+
    " - one of #{PowerSwitch::COMLIST.keys.map{|k| k.to_s}.join(', ')};\n"+
    " - one of #{PowerSwitch::POWERMODES.join(', ')} followed by the plug number." unless done
end
