#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Start Gunicorn
gunicorn --workers 2 --bind 0.0.0.0:8080 app:app
