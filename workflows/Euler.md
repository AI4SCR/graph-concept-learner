# Euler Setup

## Install

### Create virtual environment

First set the default to the new software stack by running `set_software_stack.sh new` on Euler.

```bash
mkdir -p ~/projects/ai4scr/
# use module spider <search_term> to discover available modules
module load gcc/11.4.0 python/3.8.18
python -m venv gcl
source gcl/bin/activate
```

### Upload files

```bash
rsync -ahvp ~/projects/ai4scr/ATHENA adrianom@euler:~/projects/ai4scr/
rsync -ahvp ~/projects/ai4scr/spatial-omics adrianom@euler:~/projects/ai4scr/
rsync -ahvp ~/projects/ai4scr/graph-concept-learner adrianom@euler:~/projects/ai4scr/
```

### Install packages

```bash
# Varia
pip install einops
pip install ruamel.yaml
pip install mlflow
pip3 install torch
pip install torch_geometric
pip install pre-commit
pip install "PuLP<2.8"
pip install snakemake
```

```bash
# with GPU
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.12.0+cu117.html

# without GPU
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

```bash
wget https://github.com/harelba/q/archive/refs/tags/v3.1.6.tar.gz
tar -xvf v3.1.6.tar.gz

```

```bash
pip install -e ~/projects/ai4scr/spatial-omics
pip install -e ~/projects/ai4scr/ATHENA
pip install -e ~/projects/ai4scr/graph-concept-learner
```
