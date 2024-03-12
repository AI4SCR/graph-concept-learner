import click
from graph_cl.data_models.ProjectSettings import ProjectSettings
from graph_cl.data_models.Concept import ConceptConfig

from graph_cl.data_models.Data import DataConfig
from graph_cl.data_models.Model import ModelGCLConfig, ModelGNNConfig
from graph_cl.data_models.Train import TrainConfig

"""
To include default configuration files in your package and use them in your functions, you can follow these steps:

### Step 1: Include Configuration Files in Your Package

1. Place your configuration files (e.g., YAML files) in a suitable directory within your package. For example, if your package structure looks like this:

    ```
    your_package/
    ├── __init__.py
    ├── module.py
    └── configs/
        ├── default_config1.yaml
        └── default_config2.yaml
    ```

2. If you are using setuptools for packaging, ensure your configuration files are included in your package distribution. You can do this by specifying package data in `setup.py` or `setup.cfg`. For example, in `setup.py`:

    ```python
    from setuptools import setup, find_packages

    setup(
        name="your_package",
        version="0.1",
        packages=find_packages(),
        package_data={
            "your_package": ["configs/*.yaml"],  # Include YAML files in configs/ directory
        },
        # Other parameters...
    )
    ```

    Or in `setup.cfg` under the `[options.package_data]` section:

    ```
    [options.package_data]
    your_package = configs/*.yaml
    ```

### Step 2: Access and Load Configuration Files in Your Code

To access and use these configuration files within your package, you can use the `pkg_resources` module provided by setuptools or `importlib.resources` (for Python 3.7+). Here's how you can do it with `importlib.resources` as it's part of the standard library:

```python
from importlib.resources import path, read_text
import yaml
from your_package import configs  # Import the configs package

def load_default_config(config_name):
    # Use 'path' for accessing files to open or 'read_text' to directly get the content
    config_content = read_text(configs, config_name)

    # Assuming the configuration files are YAML
    return yaml.safe_load(config_content)

# Example usage
default_config1 = load_default_config('default_config1.yaml')
print(default_config1)
```

### Important Notes:

- Replace `"your_package"` with the actual name of your package and adjust file paths according to your package structure.
- `importlib.resources.path` provides a context manager for temporary access to the resource file, which is useful if you're opening the file directly (e.g., with `open`). `importlib.resources.read_text` returns the content of the file as a string.
- This example uses `yaml.safe_load` from PyYAML to load the YAML content. Ensure you have `PyYAML` installed (`pip install pyyaml`) and adjust the loading method according to your file format.

By following these steps, you can easily include default configuration files in your package and access them in your functions, allowing for flexible and maintainable default configurations.

"""


def project():
    ps = ProjectSettings()
    ps.init()
    print(
        "Project initialized. Continue with creating a dataset with `graph_cl create dataset -n <dataset_name>`"
    )
    # note: create config files
    # TODO: dump default config file


def dataset(dataset_name: str):
    ps = ProjectSettings(dataset_name=dataset_name)
    ps.init()


def experiment(experiment_name: str):
    import yaml
    from pathlib import Path

    ps = ProjectSettings(experiment_name=experiment_name)
    ps.init()

    for config in Path(
        "/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/test/configuration"
    ).glob("*.yaml"):
        with open(config) as f:
            config_data = yaml.safe_load(f)
        with open(ps.experiment_config_dir / config.name, "w") as f:
            yaml.dump(config_data, f, sort_keys=False, indent=4)
