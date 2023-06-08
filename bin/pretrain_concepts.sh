#!/bin/bash

Help()
{
   # Display Help
   echo "Pretrains conept GNNs."
   echo
   echo "Positional arguments:"
   echo
   echo '$1    Directory where to find the concepts data. It expects every concept to have a subdirectory named `data`.'
   echo '      Likewise, this directory must contain only the concept dirs and the `configs` dir.'
   echo
   echo '$2    Name of the model that is being used. Example: 3_layer_GIN_plus_linear_layer'
   echo
}

# Get the options
while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
     \?) # incorrect option
         echo "Error: Invalid option"
         exit;;
   esac
done

### Main program ###
echo "Pretraining $(ls $1 | grep --invert-match "configs" | wc -l) concepts."

# For every concept in $1 pretrain a model in the background
for concpet in $(ls $1 | grep --invert-match "configs"); do
    # Make dir where to save the model if it does not already exist
    mkdir -p "$1/$concpet/models/$2"

    # Pretrain in the background
    echo "Lunching pretraining for $concpet $2..."
    python ./pretrain_cg_model.py "$1/$concpet/data" "$1/$concpet/models/$2" > "$1/$concpet/models/$2/model_info.txt" &
done

# Wait for all children procceses to finish
wait

echo "Done!"

# script to pre train only one concept
# Run the training with the environment
# echo "Training model..."
# python ./pretrain_cg_model.py $1 $2 > "$2/model_info.txt"
