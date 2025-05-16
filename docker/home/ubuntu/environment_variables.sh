bash
#!/bin/bash

# Set environment variables for the application.

export APP_NAME="MyApplication"
export APP_ENV="production"
export DATABASE_URL="mongodb://localhost:27017/myapp"
export REDIS_URL="redis://localhost:6379"
export SMTP_SERVER="smtp.example.com"
export EMAIL_USER="user@example.com"
export EMAIL_PASSWORD="password123"
export SECRET_KEY="yoursecretkey12345"

echo "Environment variables set."
