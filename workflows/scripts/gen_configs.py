#!/usr/bin/env python3
from itertools import product
import uuid
import yaml
import os
import sys

(prog_name, base_config_path, out_dir) = sys.argv
# base_config_path = "/Users/ast/Documents/GitHub/datasets/jackson/prediction_targets/ERStatus/configs/base_configs/pretrain_models_base_config.yaml"
# pred_target = "ERStatus"
# out_dir = "/Users/ast/Documents/GitHub/datasets/jackson/prediction_targets/ERStatus/configs/model_configs"

# Make output dir.
os.makedirs(out_dir, exist_ok=True)

# Load base config
with open(base_config_path) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    cfg = yaml.load(file, Loader=yaml.Loader)  # TODO: what is the Loader kwarg?

# Make dot product
keys = [key for key in cfg]
options = [cfg[key] for key in cfg]
prod = product(*options)

# Create new config and save to file under the mlflow run id
for i, tup in enumerate(prod):
    # Make new config
    new_cfg = dict(zip(keys, tup))

    # Hash config to get id
    cfg_id = uuid.uuid4()

    # new_cfg = list(new_cfg)
    path_new_config = os.path.join(out_dir, f"{cfg_id}.yaml")

    # Check if config file with this name already exists.
    while os.path.exists(path_new_config):
        cfg_id = uuid.uuid4()
        path_new_config = os.path.join(out_dir, f"{cfg_id}.yaml")

    # Write config
    with open(path_new_config, "w") as file:
        yaml.dump(new_cfg, file, default_flow_style=False)
