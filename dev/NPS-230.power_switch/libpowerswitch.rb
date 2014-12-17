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
      @connection.cmd(pwd)
    end

    def close
      @connection.cmd("/X")
      @connection.close
    end

    def cmd(comlist)
      case comlist
      when NilClass
        nil
      when Array
        comlist.map{ |c| self.cmd(c) }
      when String
        @connection.cmd(comlist) { |c| print c }
      else
        raise RuntimeError,"Can only deal with input argument 'comlist' as an " +
          "Array, NilClass or Strings, not class '#{comlist.class}'." unless comlist.is_a?(Array)
      end
      return self
    end

    def help
      cmd("/H")
    end

    def status
      cmd("/S")
    end

    def status
      self.cmd("")
    end

  end
end

#run status if this file is called explicitly
if __FILE__==$0
  puts PowerSwitch::Control.new.status.close
end