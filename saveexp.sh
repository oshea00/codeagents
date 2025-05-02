#!/bin/bash

# Display help message if no argument is provided
if [[ -z "$1" || "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: $0 <experiment_name>"
  echo "This utility copies all the artifacts (Dockerfile, problem.txt, requirements.txt, ./src) "
  echo "to a target directory named 'src_<experiment_name>'."
  echo "Example: $0 experiment_01"
  echo "saves to ./src_experiment_01"
  exit 1
fi

# Copy artifacts to the target directory
cp Dockerfile src/
cp problem.txt src/
cp requirements.txt src/
mv src src_$1