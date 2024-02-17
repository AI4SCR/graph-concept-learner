import os
from pathlib import Path
import yaml
from .configurator import Configuration

# %%
# def load_config(config_path: str | Path):
#     with open(config_path, "r") as stream:
#         try:
#             config = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#
#     return Configuration(**config)
#
#
# config_path = os.getenv("CONFIGURATION")
# CONFIG = load_config(config_path)
