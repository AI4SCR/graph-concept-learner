#!/usr/bin/env python3

import sys
import yaml
import os

# Unpack arguments
input_dir = sys.argv[1]
output_file = sys.argv[2]

# If exclude is not empty
if len(sys.argv) > 3:
    exclude = sys.argv[3:]
else:
    exclude = []

# Get paths to all concept configs
cfgs = [
    os.path.join(input_dir, cfg)
    for cfg in os.listdir(input_dir)
    if cfg.endswith(".yaml")
]

# Instantiate the output dictionary
concept_set = {}

for cfg_path in cfgs:
    # Get concept name
    concept = os.path.basename(cfg_path).split(".")[0]

    if concept in exclude:
        continue

    # Open config
    with open(cfg_path) as file:
        cfg = yaml.load(file, Loader=yaml.Loader)

    # Write to output
    concept_set[concept] = cfg

# Write concept-set to output
with open(output_file, "w") as file:
    yaml.dump(concept_set, file, default_flow_style=False)
