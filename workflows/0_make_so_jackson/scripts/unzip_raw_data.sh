#!/bin/bash

# cd into raw_data_dir
cd $1

# Unzip OMEandSingleCellMasks.zip
unzip -j zipped/OMEandSingleCellMasks.zip OMEnMasks/Basel_Zuri_masks.zip -d ./unzipped/OMEnMasks
unzip ./unzipped/OMEnMasks/Basel_Zuri_masks.zip -d ./unzipped/OMEnMasks
rm ./unzipped/OMEnMasks/Basel_Zuri_masks.zip

# Unzip singlecell_locations.zip
unzip -j zipped/singlecell_locations.zip \
    Zurich_SC_locations.csv \
    Basel_SC_locations.csv \
    -d ./unzipped/singlecell_locations

# Unzip singlecell_cluster_labels.zip
unzip zipped/singlecell_cluster_labels.zip -d ./unzipped

# Unzip Data_publication.zip
unzip -j zipped/SingleCell_and_Metadata.zip \
    Data_publication/Basel_Zuri_StainingPanel.csv \
    -d ./unzipped/Data_publication
unzip -j zipped/SingleCell_and_Metadata.zip \
    Data_publication/BaselTMA/Basel_PatientMetadata.csv \
    Data_publication/BaselTMA/SC_dat.csv \
    -d ./unzipped/Data_publication/BaselTMA
unzip -j zipped/SingleCell_and_Metadata.zip \
    Data_publication/ZurichTMA/Zuri_PatientMetadata.csv \
    Data_publication/ZurichTMA/SC_dat.csv \
    -d ./unzipped/Data_publication/ZurichTMA
