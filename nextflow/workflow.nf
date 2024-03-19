#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.EXPERIMENT_NAME = 'test'
params.BASE_DIR = '/Users/adrianomartinelli/data/ai4src/graph-concept-learner-test/'
params.devRun = true

// Process Definitions
process Setup {
    publishDir "results", mode: 'copy'

    output:
    path "setup_done.txt" // Dummy output to signify the process completion

    script:
    """
    graph_cl create project
    graph_cl create experiment -e ${params.EXPERIMENT_NAME}
    touch setup_done.txt
    """
}

process CreateConceptGraphs {
    publishDir "results", mode: 'copy'

    input:
    tuple val(sample_name), val(concept_name)

    output:
    path "${sample_name}_${concept_name}_graph.txt" // Dummy output for demonstration

    script:
    """
    graph_cl experiment create-concept-graph -e ${params.EXPERIMENT_NAME} -s "$sample_name" -c "$concept_name"
    touch ${sample_name}_${concept_name}_graph.txt
    """
}

process Preprocess {
    publishDir "results", mode: 'copy'

    output:
    path "preprocess_done.txt"

    script:
    """
    graph_cl experiment preprocess -e ${params.EXPERIMENT_NAME}
    touch preprocess_done.txt
    """
}

process Pretrain {
    publishDir "results", mode: 'copy'

    output:
    path "pretrain_done.txt"

    script:
    """
    graph_cl experiment pretrain -e ${params.EXPERIMENT_NAME} -c concept_1
    graph_cl experiment pretrain -e ${params.EXPERIMENT_NAME} -c concept_2
    touch pretrain_done.txt
    """
}

process Train {
    publishDir "results", mode: 'copy'

    output:
    path "train_done.txt"

    script:
    """
    graph_cl experiment train -e ${params.EXPERIMENT_NAME}
    touch train_done.txt
    """
}

// Main workflow definition
workflow {
    // Define and collect sample and concept names
    def sample_names = Channel.fromPath("/Users/adrianomartinelli/data/ai4src/graph-concept-learner/01_datasets/jackson/04_samples/*.json")
                            .map { file -> file.baseName.replace(".json", "") }
                            .view { "Sample names: $it" }

    def concept_names = Channel.fromPath("/Users/adrianomartinelli/data/ai4src/graph-concept-learner/03_concepts/*.yaml")
                            .map { it.baseName.replace(".yaml", "") }
                            .view { "Concept names: $it" }

    // Combine sample names and concept names
    sample_concept_pairs = sample_names.combine(concept_names)

    // Workflow steps
    setup_done = Setup()

    sample_concept_pairs | CreateConceptGraphs

    preprocess_done = Preprocess()
    pretrain_done = Pretrain()
    train_done = Train()
}
