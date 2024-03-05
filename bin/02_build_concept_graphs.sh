#!/bin/env python

# build single graph
#MASK_PATH="$HOME/data/ai4src/graph-concept-learner/test/02_processed/masks/BaselTMA_SP42_145_X2Y4_142.tiff"
#LABELS_PATH="$HOME/data/ai4src/graph-concept-learner/test/02_processed/labels/observations/BaselTMA_SP42_145_X2Y4_142.parquet"
#CONFIG_PATH="$HOME/data/ai4src/graph-concept-learner/data/00_concepts/radius_tumor_immune.yaml"
#OUTPUT_PATH="$HOME/data/ai4src/graph-concept-learner/test/03_concept_graphs/"
#python -m cli.main preprocess build-concept-graph "$MASK_PATH" "$LABELS_PATH" "$CONFIG_PATH" "OUTPUT_PATH"

# build all graphs for a concept
DATA_DIR="$HOME/data/ai4src/graph-concept-learner/data"
CONCEPT_NAME='concept_1'
CONCEPT_CONFIG_PATH="$HOME/data/ai4src/graph-concept-learner/data/00_concepts/concepts.yaml"
while IFS= read -r -d '' MASK_PATH
do
  LABELS_PATH="$DATA_DIR/02_processed/labels/observations/$(basename "$MASK_PATH" .tiff).parquet"
  OUTPUT_PATH="$DATA_DIR/03_concept_graphs/$CONCEPT_NAME/$(basename "$MASK_PATH" .tiff).pt"
  python -m graph_cl.cli.main preprocess build-concept-graph "$CONCEPT_NAME" "$MASK_PATH" "$LABELS_PATH" "$CONCEPT_CONFIG_PATH" "$OUTPUT_PATH"
done< <(find "$DATA_DIR/02_processed/masks/" -type f -name "*.tiff" -print0)

# TODO: use the samples.csv file to build all graphs for all concepts
#while IFS=, read -r samples; do
#    # Execute your Python program with columns A and B as arguments
#    echo "$samples"
#done < "$HOME/data/ai4src/graph-concept-learner/data/02_processed/samples.csv"
