DATA_DIR=/Users/adrianomartinelli/data/ai4src/graph-concept-learner
DATASET_NAME=jackson
DATASET_PATH=$DATA_DIR/datasets/$DATASET_NAME
CONCEPTS_DIR=$DATA_DIR/concepts

# process dataset
python -m graph_cl.cli.main dataset process -d "$DATASET_NAME"

# create concept graph for each sample and concept

for sample_file in "$DATASET_PATH"/02_processed/samples/*.json;
do
  sample_name=$(basename $sample_file .json)
  for concept_file in "$CONCEPTS_DIR"/*.yaml;
    do
      concept_name=$(basename $concept_file .yaml)
      python -m graph_cl.cli.main concept-graph create -d "$DATASET_NAME" -s "$sample_name" -c "$concept_name"
    done
  echo
done

# create filtered samples, encode target and split dataset
EXPERIMENT_NAME="test"
python -m graph_cl.cli.main experiment preprocess -e "$EXPERIMENT_NAME"

# pretrain concept graphs
python -m graph_cl.cli.main experiment pretrain -e "$EXPERIMENT_NAME" -c "concept_1"
python -m graph_cl.cli.main experiment pretrain -e "$EXPERIMENT_NAME" -c "concept_2"

# train gcl
python -m graph_cl.cli.main experiment train -e "$EXPERIMENT_NAME"
