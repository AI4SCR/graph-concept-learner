cd /dccstor/cpath_data/envs/
conda create --prefix ./gcl python=3.8
jbsub -interactive -require "a100" -cores 4+1 -mem "5G" -queue "x86_1h" bash
conda activate gcl

# Pytorch
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

# Spatial omics
pip install ai4scr-spatial-omics

# Snakemake
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda install snakemake

# Install q for prepocessing
brew install harelba/q/q

# Install
pip install einops
pip install ruamel.yaml
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.12.0+cu117.html
pip install torch_geometric
pip install mlflow

# insatll local packages
# - athena
# - gcl
# - pyg

# Local
pip install einops
pip install ruamel.yaml
pip3 install torch
pip install torch_geometric
pip install mlflow
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

cd graph_copncept_learner
pip install -e .
