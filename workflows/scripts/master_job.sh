#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name="m_graph_gen"
#SBATCH --mail-type=BEGIN,END,FAIL

# User specific aliases and functions
module purge
module load gcc/8.2.0 python/3.8.5
source $HOME/gcl/bin/activate
cd $HOME/graph-concept-learner-pub/workflows
snakemake gen_all_attributed_graphs
