bash
#!/bin/bash

# Script to check system status

echo "Checking system disk space:"
df -h

echo "Checking memory usage:"
free -m

echo "Checking CPU load:"
uptime

echo "Checking network connectivity status:"
ping -c 3 google.com

echo "Checking system services status:"
systemctl status

echo "System status check complete."
