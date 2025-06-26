



#!/bin/bash

# Get the PID of the process using port 8000
PID=$(lsof -ti :8000)

if [ -z "$PID" ]; then
  echo "No process is using port 8000."
else
  echo "Killing process with PID: $PID"
  kill -9 $PID
  echo "Process killed."
fi
