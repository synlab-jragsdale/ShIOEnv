#!/bin/bash

# Script to configure network settings

# Set up the network interface eth0
ifconfig eth0 up
ifconfig eth0 192.168.1.100 netmask 255.255.255.0
route add default gw 192.168.1.1 eth0

# Enable DNS
echo "nameserver 8.8.8.8" > /etc/resolv.conf
echo "nameserver 8.8.4.4" >> /etc/resolv.conf

# Restart networking services
systemctl restart networking.service

echo "Network setup completed."