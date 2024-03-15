DATA_DIR=/Users/adrianomartinelli/data/ai4src/graph-concept-learner-test
DATASET_NAME=jackson
DATASET_PATH=$DATA_DIR/datasets/$DATASET_NAME
CONCEPTS_DIR=$DATA_DIR/concepts

# setup
graph_cl create project
graph_cl create dataset -d "$DATASET_NAME"
graph_cl create experiment -e "exp_1"

# symlink raw data
for file in /Users/adrianomartinelli/data/ai4src/graph-concept-learner/jackson/raw_data/zipped/*;
do
  ln -s $file "$DATASET_PATH"/01_raw/"$(basename $file)"
done

# process dataset
graph_cl dataset process -d "$DATASET_NAME"

# create concept graph for each sample and concept

#for sample_file in "$DATASET_PATH"/02_processed/samples/*.json;
#for sample_file in $(ls "$DATASET_PATH"/02_processed/samples/BaselTMA_SP4* | tail -n 9);
for sample_file in $(ls "$DATASET_PATH"/02_processed/samples/ZTMA208_slide_* | head ; ls "$DATASET_PATH"/02_processed/samples/BaselTMA_SP4* | head -n 25);
do
  sample_name=$(basename $sample_file .json)
  echo "Creating concept graphs for sample: $sample_name"
  for concept_file in "$CONCEPTS_DIR"/*.yaml;
    do
      concept_name=$(basename $concept_file .yaml)
      graph_cl concept-graph create -d "$DATASET_NAME" -s "$sample_name" -c "$concept_name"
    done
  echo
done

# create filtered samples, encode target and split dataset
EXPERIMENT_NAME="test"
graph_cl experiment preprocess -e "$EXPERIMENT_NAME"

# pretrain concept graphs, note: we explicitly define the concepts that are in the model_gcl.yaml
graph_cl experiment pretrain -e "$EXPERIMENT_NAME" -c "concept_1"
graph_cl experiment pretrain -e "$EXPERIMENT_NAME" -c "concept_2"

# train gcl
graph_cl experiment train -e "$EXPERIMENT_NAME"
