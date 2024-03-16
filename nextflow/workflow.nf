#!/usr/bin/env nextflow

params.data_dir = '/Users/adrianomartinelli/data/ai4src/graph-concept-learner-test-2'
params.dataset_name = 'jackson'
params.concepts_dir = "$params.data_dir/concepts"
params.experiment_name = "exp_1"

process Setup {
    script:
    """
    graph_cl create project
    graph_cl create dataset -d "${params.dataset_name}"
    graph_cl create experiment -e "${params.experiment_name}"
    """
}

# TODO: remove
process SymlinkRawData {
    script:
    """
    mkdir -p ${params.data_dir}/datasets/${params.dataset_name}/01_raw
    for file in /Users/adrianomartinelli/data/ai4src/graph-concept-learner/jackson/raw_data/zipped/*;
    do
      ln -s \$file ${params.data_dir}/datasets/${params.dataset_name}/01_raw/\$(basename \$file)
    done
    """
}

process ProcessDataset {
    script:
    """
    graph_cl dataset process -d "${params.dataset_name}"
    """
}

process CreateConceptGraphs {
    input:
    tuple val(sample_name), path(sample_file), path(concept_file)

    script:
    """
    concept_name=\$(basename \$concept_file .yaml)
    graph_cl concept-graph create -d "${params.dataset_name}" -s "\$sample_name" -c "\$concept_name"
    """
}

process PreprocessExperiment {
    script:
    """
    graph_cl experiment preprocess -e "${params.experiment_name}"
    """
}

process PretrainConceptGraphs {
    script:
    """
    graph_cl experiment pretrain -e "${params.experiment_name}" -c "concept_1"
    graph_cl experiment pretrain -e "${params.experiment_name}" -c "concept_2"
    """
}

process TrainGCL {
    script:
    """
    graph_cl experiment train -e "${params.experiment_name}"
    """
}

workflow {
    Setup()
    SymlinkRawData()
    ProcessDataset()

    // Prepare channels for parallel processing of concept graphs
    sample_files_ch = Channel.fromPath("${params.data_dir}/datasets/${params.dataset_name}/02_processed/samples/*.json")
    concept_files_ch = Channel.fromPath("${params.concepts_dir}/*.yaml").toList() // Convert to list to broadcast to each sample

    // Combine each sample file with all concept files for parallel processing
    sample_concept_comb = sample_files_ch.combine(concept_files_ch)

    // Process each sample with each concept in parallel
    sample_concept_comb.map { sample_file, concept_files ->
        sample_name = sample_file.baseName
        concept_files.collect { concept_file ->
            tuple(sample_name, sample_file, concept_file)
        }
    }.set { graphs_to_create }
    CreateConceptGraphs(graphs_to_create.flatten())

    PreprocessExperiment()
    PretrainConceptGraphs()
    TrainGCL()
}
