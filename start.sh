#!/bin/bash
export PYTHONSAFEPATH=1
export PYTHONPATH=/app/.venv/lib/python3.11/site-packages:$PYTHONPATH
exec gunicorn --worker-tmp-dir /dev/shm app:app
