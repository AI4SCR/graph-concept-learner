#!/usr/bin/env python3

import os
import sys
from snakemake.utils import read_job_properties

# Get job script
jobscript = sys.argv[1]

# Read job properties
job_properties = read_job_properties(jobscript)
cores = job_properties["resources"]["cores"]
mem = job_properties["resources"]["mem"]
queue = job_properties["resources"]["queue"]
log_base_path = job_properties["log"][0]
rule_name = job_properties["rule"]

# Get time
if queue == "x86_1h":
    time="1:00:00"
elif queue == "x86_6h":
    time="6:00:00"
elif queue == "x86_24h":
    time="24:00:00"
else:
    assert False, "Run time not well specified"

if "+" in cores:
    cpus = cores.split("+")[0]
    gpus = cores.split("+")[1]
else:
    cpus = cores
    gpus = "0"

# Call jobscript using batch system
os.system(
    f'sbatch --job-name="{rule_name}" --cpus-per-task={cpus} --gpus={gpus} mem-per-cpu={mem} --time={time} --output="{log_base_path}.stdout" --error="{log_base_path}.stderr" {jobscript}'
)