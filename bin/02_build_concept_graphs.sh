#!/bin/env python

# build single graph
#MASK_PATH="$HOME/data/ai4src/graph-concept-learner/test/02_processed/masks/BaselTMA_SP42_145_X2Y4_142.tiff"
#LABELS_PATH="$HOME/data/ai4src/graph-concept-learner/test/02_processed/labels/observations/BaselTMA_SP42_145_X2Y4_142.parquet"
#CONFIG_PATH="$HOME/data/ai4src/graph-concept-learner/data/00_concepts/radius_tumor_immune.yaml"
#OUTPUT_PATH="$HOME/data/ai4src/graph-concept-learner/test/03_concept_graphs/"
#python -m cli.main preprocess build-concept-graph "$MASK_PATH" "$LABELS_PATH" "$CONFIG_PATH" "OUTPUT_PATH"

# build all graphs for a concept
DATA_DIR="$HOME/data/ai4src/graph-concept-learner/test"
MASK_DIR="$DATA_DIR/02_processed/masks"
CONCEPT_CONFIG_PATH="$HOME/data/ai4src/graph-concept-learner/data/00_concepts/radius_tumor_immune.yaml"
OUTPUT_DIR="$DATA_DIR/03_concept_graphs/$(basename "$CONCEPT_CONFIG_PATH" .yaml)"
while IFS= read -r -d '' MASK_PATH
do
  LABELS_PATH="$DATA_DIR/02_processed/labels/observations/$(basename "$MASK_PATH" .tiff).parquet"
  echo python -m cli.main preprocess build-concept-graph "$MASK_PATH" "$LABELS_PATH" "$CONFIG_PATH" "$OUTPUT_DIR"
done < <(find "$HOME/data/ai4src/graph-concept-learner/test/02_processed/masks/" -type f -name "*.tiff")


find $MASK_DIR -type f -name "*.tiff"
