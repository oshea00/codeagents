#!/bin/bash

# Display help message if no argument is provided or if --help/-h is passed
if [[ -z "$1" || "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: $0 <directory>"
  echo "Analyze metrics for the specified directory containing main.py."
  echo "Creates metrics.txt report in the target directory."
  echo "Example: $0 ./src"
  exit 1
fi

echo "Analyzing metrics for directory: $1" > $1/metrics.txt
radon cc -a $1/main.py | grep "Average complexity" >> $1/metrics.txt
echo "Maintainability: " $(radon mi $1/main.py) >> $1/metrics.txt
radon raw $1/main.py | grep "LLOC:" >> $1/metrics.txt
radon hal $1/main.py | grep -E "difficulty|time|bugs" >> $1/metrics.txt