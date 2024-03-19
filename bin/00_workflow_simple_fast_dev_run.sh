PROJECT_BASE_DIR=/Users/adrianomartinelli/data/ai4src/graph-concept-learner
DATASET_NAME=jackson
DATASET_PATH=/Users/adrianomartinelli/.cache/ai4bmr/datasets/$DATASET_NAME
CONCEPTS_DIR=$PROJECT_BASE_DIR/03_concepts
EXPERIMENT_NAME="exp_0"

# setup
graph_cl create project
graph_cl create experiment -e "$EXPERIMENT_NAME"

# symlink raw data
for file in /Users/adrianomartinelli/data/ai4src/graph-concept-learner/jackson/raw_data/zipped/*;
do
  ln -s $file "$DATASET_PATH"/01_raw/"$(basename $file)"
done

# process dataset
graph_cl dataset process -d "$DATASET_NAME"

# create concept graph for each sample and concept
for sample_file in $(ls "$DATASET_PATH"/04_samples/ZTMA208_slide_* | head ; ls "$DATASET_PATH"/04_samples/BaselTMA_SP4* | head -n 25);
#for sample_file in "$DATASET_PATH"/02_processed/samples/*.json;
do
  sample_name=$(basename $sample_file .json)
  printf "\e[33mProcess\e[0m sample: %s\n" "$sample_name"
  printf "\e[32mCreated\e[0m graph for: "
  for concept_file in "$CONCEPTS_DIR"/*.yaml;
    do
      concept_name=$(basename $concept_file .yaml)
      graph_cl experiment create-concept-graph -e "$EXPERIMENT_NAME" -s "$sample_name" -c "$concept_name"
      printf "%s ✅ " $concept_name
    done
  echo
done

# create filtered samples, encode target and split dataset
graph_cl experiment preprocess -e "$EXPERIMENT_NAME"

# pretrain concept graphs, note: we explicitly define the concepts that are in the model_gcl.yaml
graph_cl experiment pretrain -e "$EXPERIMENT_NAME" -c "concept_1"
graph_cl experiment pretrain -e "$EXPERIMENT_NAME" -c "concept_2"

# train gcl
graph_cl experiment train -e "$EXPERIMENT_NAME"
