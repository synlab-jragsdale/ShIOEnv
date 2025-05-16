bash
#!/bin/bash

# Auto repair script for system services

echo "Starting automatic system repair..."

# Check for failed services
failed_services=$(systemctl --failed)
if [ -z "$failed_services" ]; then
    echo "No failed services detected."
else
    echo "Attempting to restart failed services..."
    for service in $(echo $failed_services | awk '{print $2}'); do
        echo "Restarting $service..."
        systemctl restart $service
        if [ $? -eq 0 ]; then
            echo "$service restarted successfully."
        else
            echo "Failed to restart $service."
        fi
    done
fi

# Clean up any orphaned packages
echo "Cleaning up orphaned packages..."
sudo apt-get autoremove -y

# Repair and configure packages
echo "Repairing and configuring any problematic packages..."
sudo dpkg --configure -a
sudo apt-get install -f

echo "Auto repair process completed."
