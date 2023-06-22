#!/bin/bash

# unpack arguments
URLS=$1
ROOT=$2

# cd into root directory
cd $ROOT

# make dir for log files
mkdir -p download_info/download_logs

# make new dir for downloads and cd into it
mkdir zipped
cd zipped

# Download each file on a separate process and log
cat $URLS | xargs -n 2 -P 3 wget -o
