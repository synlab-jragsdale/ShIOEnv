bash
#!/bin/bash

# Maintenance script for routine server checkup

echo "Starting server maintenance operations."

# Update all packages
echo "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Clean up unnecessary files
echo "Cleaning up disk space..."
sudo apt-get autoremove -y
sudo apt-get autoclean

# Check disk space usage
echo "Checking disk space..."
df -h

# Restarting services
echo "Restarting essential services..."
sudo systemctl restart apache2
sudo systemctl restart mysql

echo "Maintenance operations completed."

# Log maintenance activity
echo "$(date '+%Y-%m-%d %H:%M:%S') - Maintenance run complete" >> /var/log/maintenance_log.log

exit 0
