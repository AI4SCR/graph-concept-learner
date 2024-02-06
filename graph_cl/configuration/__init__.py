from pathlib import Path
import yaml
from .configurator import Configuration

# %%
config_path = Path("/configs/config.yml")


def load_config(config_path: str):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return Configuration(**config)


CONFIG = load_config(
    "/Users/adrianomartinelli/projects/ai4scr/graph-concept-learner/configs/config.yml"
)
