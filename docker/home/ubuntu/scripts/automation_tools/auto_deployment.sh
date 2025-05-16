#!/bin/bash

# Auto Deployment Script for Application Services

echo "Starting the automated deployment process..."
date

# Fetch the latest code from the repository
echo "Updating code repository..."
git pull origin master

# Build and compile the project
echo "Building the project..."
make clean install

# Perform tests
echo "Running tests..."
make test

# Deploy to staging
echo "Deploying to staging environment..."
scp -r ./build staging@staging.server.com:/var/www/html/

# Verify staging deployment
echo "Verifying deployment on staging..."
curl -s http://staging.server.com/ | grep "Success"

# Deploy to production after confirmation
read -p "Deploy to production? (y/n): " answer
if [ "$answer" == "y" ]; then
    echo "Deploying to production environment..."
    scp -r ./build prod@production.server.com:/var/www/html/
    echo "Deployment to production completed."
else
    echo "Deployment aborted by user."
fi

echo "Deployment process completed