#!/bin/env python
DATA_DIR="$HOME/data/ai4src/graph-concept-learner/test/"
EXPERIMENT_DIR="$HOME/data/ai4src/graph-concept-learner/experiments/ERStatusV2"
python -m cli.main preprocess attribute-graphs $EXPERIMENT_DIR $DATA_DIR

# RUN PRETRAINING FOR ONE FOLD AND ONE CONCEPT
folds_dir="/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds"
fold_path="/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds/fold_0"
model_config_path="/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model_gnn.yaml"
train_config_path="/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/pretrain.yaml"
python -m cli.main model pretrain "radius_tumor_immune" $fold_path $model_config_path $train_config_path

# RUN PRETRAINING FOR ALL FOLDS AND ALL CONCEPTS
folds_dir="/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds"
model_config_path="/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model_gnn.yaml"
train_config_path="/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/pretrain.yaml"
# note: with `find`, for loops are fragile and can break if the file names contain spaces or newlines. Use `while` loops with `read` instead.
while IFS= read -r -d '' fold_path
do
  for concept in "$fold_path"/attributed_graphs/*; do
    python -m cli.main model pretrain "$(basename "$concept")" "$fold_path" "$model_config_path" "$train_config_path"
  done
done <   <(find -E $folds_dir -type d -regex ".*fold_[0-9]+" -print0)

# RUN TRAINING FOR ONE FOLD AND ONE CONCEPT
fold_path="/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/data/05_folds/fold_0"
train_config_path="/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/train.yaml"
model_gcl_config_path="/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model_gcl.yaml"
model_gnn_config_path="/Users/adrianomartinelli/data/ai4src/graph-concept-learner/experiments/ERStatusV2/configuration/model_gnn.yaml"
python -m graph_cl.cli.main model train "$fold_path" "$train_config_path" "$model_gcl_config_path" "$model_gnn_config_path"

# RUN TRAINING FOR ALL FOLDS AND ALL CONCEPTS
while IFS= read -r -d '' fold_path
do
  python -m graph_cl.cli.main model train "$fold_path" "$train_config_path" "$model_gcl_config_path" "$model_gnn_config_path"
done <  <(find -E "$folds_dir" -type d -regex "folds_[0-9]+" -print0)
