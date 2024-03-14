DATA_DIR=/Users/adrianomartinelli/data/ai4src/graph-concept-learner
DATASET_NAME=jackson
DATASET_PATH=$DATA_DIR/datasets/$DATASET_NAME
CONCEPTS_DIR=$DATA_DIR/concepts

# process dataset
python -m graph_cl.cli.main dataset process -d "$DATASET_NAME"

# create concept graph for each sample and concept
counter=0
for sample_file in "$DATASET_PATH"/02_processed/samples/*.json;
do
  sample_name=$(basename $sample_file .json)
  printf "\e[33mProcess\e[0m sample: %s\n" "$sample_name"
  accumulated_concepts=""
  for concept_file in "$CONCEPTS_DIR"/*.yaml;
    do
      concept_name=$(basename $concept_file .yaml)
      python -m graph_cl.cli.main concept-graph create -d "$DATASET_NAME" -s "$sample_name" -c "$concept_name"
      if [ -z "$accumulated_concepts" ]; then
        accumulated_concepts="$concept_name ✅"
    else
        accumulated_concepts="$accumulated_concepts, $concept_name ✅"
    fi
      printf "\r\e[32mCreated\e[0m graph for: %s" "$accumulated_concepts"
    done
  echo
  # Increment the counter
  ((counter++))

  # Break out of the loop if the counter reaches 20
  if [ "$counter" -eq 20 ]; then
    break
  fi
done
