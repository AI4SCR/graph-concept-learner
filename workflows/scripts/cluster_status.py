#!/usr/bin/env python3
import sys
import subprocess

jobid = sys.argv[1]  # This apparently doesnt provide the job id. Needed.
state = subprocess.run(f"bjobs -o 'stat' -noheader {jobid}")
map_state = {"PEND": "running", "RUN": "running", "AVAIL": "running", "DONE": "success"}

print(map_state.get(state, "failed"))
