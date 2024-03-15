# Graph Concept Learner
Learning using concept graphs for general prediction tasks.

## Installation

### Package installation
```sh
# assuming you have an SSH key set up on GitHub
pip install "git+ssh://git@github.com:AI4SCR/graph-concept-learner.git@refactoring"
```

### Suggested setup for development
```sh
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
pre-commit install
```

### CLI
Start exploring the CLI by running `graph_cl --help`

## Usage
Check workflow [Workflow tutorial](https://github.com/AI4SCR/graph-concept-learner/wiki/Workflow-tutorial)



## ToDos
- [ ] Further modularize the code:
   - [x] Create functions that just create models
   - [x] train functions just load the model and train it
   - [x] create function that create datasets
- [ ] Refactor and unify base models, remove custom transformers if possible
- [x] exclude testing from training scripts
- [ ] create plotting utilities -> some are in utils/mlflow
- [x] unify configs where possible
- [x] create configs from pydantic models
- [x] parameterize LitModules
- [ ] download dataset for easy deployment on the cloud
- [x] use relative paths within the graph_cl module
- [x] refine Pydantic models
- [x] refactor tests and complete tests
- [ ] write evaluation code
- [x] refactor ./bin
