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

# Call jobscript using batch system
os.system(
    f"jbsub -name {rule_name} -cores {cores} -mem {mem} -queue {queue} -out {log_base_path}.stdout -err {log_base_path}.stderr {jobscript}"
)
