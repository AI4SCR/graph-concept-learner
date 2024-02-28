# Graph Concept Learner

## ToDos
 - Further modularize the code:
   - Create functions that just create models
   - train functions just load the model and train it
   - create function that create datasets
 - download dataset for easy deployment on the cloud

Learning using concept graphs for general prediction tasks.

## Environment
Using a virtual environment for all commands in this guide is strongly recommended.

## Installation

### Package installation
```sh
# assuming you have an SSH key set up on GitHub
pip install "git+ssh://git@github.com:AI4SCR/graph-concept-learner.git@main"
```

### Suggested setup for development
```sh
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r dev_requirements.txt
pip install -e .
pre-commit install
```

## Usage
Check workflow [Workflow tutorial](https://github.com/AI4SCR/graph-concept-learner/wiki/Workflow-tutorial)

## Contributing

Check [CONTRIBUTING.md](./CONTRIBUTING.md).

## Getting support

Check [SUPPORT.md](./SUPPORT.md).
