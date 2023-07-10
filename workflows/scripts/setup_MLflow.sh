#!/bin/bash
# If the
if [ $1 == "True"]; then
        export MLFLOW_TRACKING_URI=http://experiments2.traduce.zc2.ibm.com:5000
        export MLFLOW_S3_ENDPOINT_URL=http://data.digital-pathology.zc2.ibm.com:9000

        export AWS_ACCESS_KEY_ID="pus@zurich.ibm.com"
        export AWS_SECRET_ACCESS_KEY="swill.varied.natty.wagtail"
        export MLFLOW_EXPERIMENT_NAME=""

        if [ -z ${AWS_SECRET_ACCESS_KEY} ]; then
                echo "WARNING: The env var AWS_SECRET_ACCESS_KEY is either blank or not set!"
        fi
else
        echo "mlflow_on_remote_server = $1. Logging run to local file system"
fi
