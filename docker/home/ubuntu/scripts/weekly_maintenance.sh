bash
#!/bin/bash

# Weekly Maintenance Script

echo "Starting weekly maintenance tasks."

# Clear system cache
echo "Clearing system cache."
sudo sync; sudo sysctl -w vm.drop_caches=3

# Update and upgrade system packages
echo "Updating system packages."
sudo apt-get update && sudo apt-get -y upgrade

# Check disk usage and free space
echo "Checking disk space."
df -h

# Remove old logs
echo "Removing old logs."
find /var/log -type f -name '*.log' -mtime +7 -exec rm {} \;

# Restart services to free up memory
echo "Restarting critical services."
sudo systemctl restart apache2
sudo systemctl restart mysql

# Execute any pending cron jobs manually
echo "Running all pending cron jobs."
run-parts /etc/cron.weekly

echo "Weekly maintenance tasks completed."
