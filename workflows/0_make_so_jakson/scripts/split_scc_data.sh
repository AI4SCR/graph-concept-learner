#!/bin/bash
BASEL_FILE=$1
ZURI_FILE=$2
OUTPUT=$3

# Make output directory
mkdir -p $OUTPUT
cd $OUTPUT

# For every cohort sort and write records to individual csv's
for file in $BASEL_FILE $ZURI_FILE; do
	sort -k 1 -t , $file |
	awk -F ',' '
		BEGIN {
			file_name="core"
		}
		{
			if(file_name==$1) {
				print > $1
			}
			else {
				close(file_name)
				file_name=$1
				print > $1
			}
		}
	'
done

# Rename with csv extension
for f in $(ls); do
	mv $f $f.csv
done
