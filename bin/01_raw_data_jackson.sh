#!/bin/env python

RAW_DIR="$HOME/data/ai4src/graph-concept-learner/data/01_raw"
PROCESSED_DIR="$HOME/data/ai4src/graph-concept-learner/test/02_processed/"

python -m cli.main data jackson "$RAW_DIR" "$PROCESSED_DIR"
