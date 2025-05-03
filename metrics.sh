#!/bin/bash

# Display help message if no argument is provided or if --help/-h is passed
if [[ -z "$1" || "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: $0 <directory> <code_coverage> <ux_score> <csv_file>"
  echo "Analyze metrics for the specified directory containing main.py."
  echo "Creates metrics.txt report in the target directory."
  echo "Appends a CSV record to the specified CSV file."
  echo ""
  echo "Parameters:"
  echo "  <directory>      Directory containing main.py"
  echo "  <code_coverage>  Code coverage percentage"
  echo "  <ux_score>       User experience score"
  echo "  <csv_file>       Path to CSV file for recording metrics"
  echo ""
  echo "Example: $0 ./src 87.5 4.2 ./metrics_history.csv"
  exit 1
fi

# Check if all required parameters are provided
if [[ -z "$2" || -z "$3" || -z "$4" ]]; then
  echo "Error: Missing required parameters."
  echo "Run '$0 --help' for usage information."
  exit 1
fi

# Assign parameters to named variables for clarity
DIRECTORY=$1
CODE_COVERAGE=$2
UX_SCORE=$3
CSV_FILE=$4

# Create the metrics.txt report
echo "Analyzing metrics for directory: $DIRECTORY" > $DIRECTORY/metrics.txt
radon cc -a $DIRECTORY/main.py | grep "Average complexity" >> $DIRECTORY/metrics.txt
echo "Maintainability: " $(radon mi $DIRECTORY/main.py) >> $DIRECTORY/metrics.txt
radon raw $DIRECTORY/main.py | grep "LLOC:" >> $DIRECTORY/metrics.txt
radon hal $DIRECTORY/main.py | grep -E "difficulty|time|bugs" >> $DIRECTORY/metrics.txt

# Extract metrics from the report
AVG_COMPLEXITY=$(grep "Average complexity" $DIRECTORY/metrics.txt | sed -E 's/.*\(([0-9.]+)\).*/\1/')
# Extract the last character of the maintainability line (the grade)
MAINTAINABILITY=$(grep "Maintainability:" $DIRECTORY/metrics.txt | awk '{print $NF}')
LLOC=$(grep "LLOC:" $DIRECTORY/metrics.txt | awk '{print $2}')
DIFFICULTY=$(grep "difficulty:" $DIRECTORY/metrics.txt | awk '{print $2}')
TIME=$(grep "time:" $DIRECTORY/metrics.txt | awk '{print $2}')
BUGS=$(grep "bugs:" $DIRECTORY/metrics.txt | awk '{print $2}')

# Create CSV header if the file doesn't exist or is empty
if [[ ! -f "$CSV_FILE" ]] || [[ ! -s "$CSV_FILE" ]]; then
  echo "directory_name,LLOC,difficulty,time,bugs,avg_complexity,maintainability_score,code_coverage,ux_score" > "$CSV_FILE"
fi

# Append CSV record
echo "$DIRECTORY,$LLOC,$DIFFICULTY,$TIME,$BUGS,$AVG_COMPLEXITY,$MAINTAINABILITY,$CODE_COVERAGE,$UX_SCORE" >> "$CSV_FILE"

echo "Metrics recorded in $DIRECTORY/metrics.txt"
echo "CSV record appended to $CSV_FILE"