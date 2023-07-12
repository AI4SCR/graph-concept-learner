#!/usr/bin/env python3
from ruamel import yaml
import os
import sys

(prog_name, path_to_main_config) = sys.argv

# Load main config
with open(path_to_main_config) as file:
    cfg = yaml.load(file, Loader=yaml.Loader)

# Make directories
# zipped
p = os.path.join(cfg["root"], "raw_data", "zipped")
os.makedirs(p, exist_ok=True)

# unzipped
p = os.path.join(cfg["root"], "raw_data", "unzipped")
os.makedirs(p, exist_ok=True)

# configs
extended_root = os.path.join(
    cfg["root"],
    "prediction_tasks",
    cfg["prediction_target"],
    f"normalized_with_{cfg['normalize_with']}",
    "configs",
)

# att_cfg
p = os.path.join(extended_root, "attribute_configs")
os.makedirs(p, exist_ok=True)

# base configs
p = os.path.join(extended_root, "base_configs")
os.makedirs(p, exist_ok=True)

# dataset configs
p = os.path.join(extended_root, "dataset_configs")
os.makedirs(p, exist_ok=True)

print(f"Folder structure in {cfg['root']} has been created.")
