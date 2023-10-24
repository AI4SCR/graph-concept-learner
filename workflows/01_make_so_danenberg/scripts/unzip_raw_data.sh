#!/bin/bash

# cd into raw_data_dir
cd $1

# Make dirs
mkdir unzipped
cd unzipped

# Unzip files into unzipped directory
for file in ../zipped/*.zip; do
    # Unzip file
    unzip $file
done

# Put loose files into singlecell_locations dir
mkdir singlecell_locations
mv *.csv singlecell_locations

# Second round of unziping
for file in $(find . -name "*.zip"); do
    # Get dir
    dir=$(dirname $file)

    # Unzip file
    unzip -d $dir $file

    # Remove file
    rm $file
done
