# Graph Concept Learner

Learning using concept graphs for general prediction tasks.

## Environment
Make sure your in the root of the project.

```bash
python3.11 -m venv ~/.venv/graph_cl
source ~/.venv/graph_cl/bin/activate
```
If you do not want to use the workflow defaults, you can set the environment variables in a `.env` file.

- AI4BMR_CACHE_DIR: The directory where the datasets are stored. This is used by the workflow to download the datasets.
  Defaults to `~/.cache/ai4bmr/`.

```bash
echo "AI4BMR_CACHE_DIR=<PATH_TO_DATASET_CACHE>" >> .env
```

## Installation

### Package installation

```sh
# assuming you have an SSH key set up on GitHub
pip install "git+ssh://git@github.com:AI4SCR/graph-concept-learner.git@refactoring"
```

### Suggested setup for development

```sh
pip install "git+https://github.com/AI4SCR/ai4bmr-core.git@main#egg=ai4bmr-core"
# for private repositories
pip install "git+ssh://git@github.com/AI4SCR/ai4bmr-core.git@main#egg=ai4bmr-core"

# or for development
git clone git+ssh://git@github.com/AI4SCR/ai4bmr-core.git
git clone git+ssh://git@github.com:AI4SCR/graph-concept-learner.git@refactoring
cd ai4bmr-core
pip install -e ".[dev, test]"
pre-commit install
```


## CLI

Start exploring the CLI by running `graph_cl --help`

## Run the Workflow

### Nextflow

```bash
wget -qO- https://get.nextflow.io | bash
chmod +x nextflow
mv nextflow /usr/local/bin
```

```bash
nextflow run workflow.nf \
--data_dir "/new/data/dir" \
--dataset_name "new_dataset" --experiment_name "exp_2" \
-log "nextflow/logs/nextflow.log"
```

### SnakeMake

```bash
snakemake --cores [N]
```

## Usage

Check workflow [Workflow tutorial](https://github.com/AI4SCR/graph-concept-learner/wiki/Workflow-tutorial)

## ToDos

- [ ] Dataset design is still flawed, we need to deconstruct the model loading from the path factory. For example to
  get the path of a sample, the current implementation requires to load all samples. This is not efficient.
  We could introduce an explicit load call, add a prevent_load flag or add the path definitions as MixIn to the dataset
  class (which could be very convoluted).
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
