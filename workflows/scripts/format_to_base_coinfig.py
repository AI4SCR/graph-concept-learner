#!/usr/bin/env python3
from ruamel import yaml
import sys

(prog_name, base_config_path) = sys.argv

# Make output dir.
with open(base_config_path) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    cfg = yaml.load(file, Loader=yaml.Loader)  # TODO: what is the Loader kwarg?

# Make everything as list
for key, value in cfg.items():
    cfg[key] = [value]

# Set seeds
cfg["seed"] = list(range(0, 100))

# Write config
with open(base_config_path, "w") as file:
    yaml.dump(cfg, file, Dumper=yaml.RoundTripDumper)
