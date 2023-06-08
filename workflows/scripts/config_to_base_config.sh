#!/usr/bin/env bash
concept=$(echo $1 | cut -d' ' -f1)
path_to_config=$(echo $1 | cut -d' ' -f2)
path_to_bases="/dccstor/cpath_data/datasets/GCL/jakson/prediction_tasks/ERStatus/normalized_with_min_max/split_basel_leave_zurich_as_external/configs/base_configs/best_"
cp -v $path_to_config $path_to_bases$concept.yaml
