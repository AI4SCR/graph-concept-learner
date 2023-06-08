##### Helper functions #####
def get_all_graphs_and_datasets(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/dataset_configs"
    CONCEPT_NAMES = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]

    CONTACT_CONCEPTS = []
    RADIUS_CONCEPTS = []
    for concept in CONCEPT_NAMES:
        if "contact" in concept:
            CONTACT_CONCEPTS.append(concept)
        else:
            RADIUS_CONCEPTS.append(concept)

    path_to_filtered_sample_ids = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/filtered_sample_ids_and_labels.csv"
    SMAPLE_IDS = list(pd.read_csv(path_to_filtered_sample_ids, index_col=0).squeeze("columns").index.values)

    all_graphs = expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/processed_data/{{concept}}/{{spl_id}}.pt", concept=CONTACT_CONCEPTS, spl_id=SMAPLE_IDS)
    all_datasets = expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/processed_data/{{concept}}", concept=RADIUS_CONCEPTS)

    return all_graphs + all_datasets

def get_all_graphs_for_a_concept(wildcards):
    path_to_filtered_sample_ids = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/filtered_sample_ids_and_labels.csv"
    SMAPLE_IDS = list(pd.read_csv(path_to_filtered_sample_ids, index_col=0).squeeze("columns").index.values)
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/processed_data/" + "{{concept}}/{spl_id}.pt", spl_id=SMAPLE_IDS)

def get_paths_to_pretrained_models(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/dataset_configs"
    CONCEPT_NAMES = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/pretrain_model_configs"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/{{concept}}/{{cfg_id}}/best_val_{{metric_name}}.pt", concept=CONCEPT_NAMES, cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_paths_to_dataset_configs(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/dataset_configs"
    CONCEPT_NAMES = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{path_to_file}/{{concept}}.yaml", concept=CONCEPT_NAMES)

def gen_paths_to_train_conf_matrices(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/train_model_configs"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/graph_concept_learners/"+"ER/{config_id}/test_conf_mat_from_best_val_{metric_name}.png", metric_name=config["follow_this_metrics"], config_id=CFG_IDS)

def gen_paths_to_space_gm_conf_matrices(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/space_gm_train_configs"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    path_to_file = f"{root}/intermediate_data/non_spatial_baseline/"
    DATASET_NAMES = [f for f in os.listdir(path_to_file)]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/space_gm_baseline/"+"{dataset}/{cfg_id}/test_conf_mat_from_best_val_weighted_f1_score.png", cfg_id=CFG_IDS, dataset=DATASET_NAMES)

def get_all_rand_graphs_and_datasets(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/dataset_configs"
    CONCEPT_NAMES = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]

    CONTACT_CONCEPTS = []
    RADIUS_CONCEPTS = []
    for concept in CONCEPT_NAMES:
        if "contact" in concept:
            CONTACT_CONCEPTS.append(concept)
        else:
            RADIUS_CONCEPTS.append(concept)

    path_to_filtered_sample_ids = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/meta_data/filtered_sample_ids_and_labels.csv"
    SMAPLE_IDS = list(pd.read_csv(path_to_filtered_sample_ids, index_col=0).squeeze("columns").index.values)

    all_graphs = expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/randomized_data/{{concept}}/{{spl_id}}.pt", concept=CONTACT_CONCEPTS, spl_id=SMAPLE_IDS)
    all_datasets = expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/randomized_data/{{concept}}", concept=RADIUS_CONCEPTS)

    return all_graphs + all_datasets

### Helpers for random ER concepts ###
def get_rnd_all_cells_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_all_cells_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/all_cells_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_all_cells_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_all_cells_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/all_cells_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_endothelial_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_endothelial_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/endothelial_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_endothelial_stromal_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_endothelial_stromal_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/endothelial_stromal_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_endothelial_tumor_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_endothelial_tumor_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/endothelial_tumor_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_immune_endothelial_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_immune_endothelial_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/immune_endothelial_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_immune_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_immune_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/immune_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_immune_stromal_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_immune_stromal_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/immune_stromal_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_immune_tumor_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_immune_tumor_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/immune_tumor_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_stromal_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_stromal_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/stromal_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_stromal_tumor_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_stromal_tumor_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/stromal_tumor_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_tumor_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_tumor_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/randomized_data/tumor_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

### Helpers for seed concepe pretraining ###
def get_best_spaceGM_ER(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_spaceGM_ER"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/space_gm_baseline/"+"best_spaceGM_ER/frequency_dataset/{cfg_id}/test_conf_mat_from_best_val_weighted_f1_score.png", cfg_id=CFG_IDS)

def get_best_spaceGM_ERless(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_spaceGM_ERless"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/space_gm_baseline/"+"best_spaceGM_ERless/frequency_dataset_ERless/{cfg_id}/test_conf_mat_from_best_val_weighted_f1_score.png", cfg_id=CFG_IDS)

def get_best_all_cells_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_all_cells_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/all_cells_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_all_cells_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_all_cells_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/all_cells_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_all_cells_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_all_cells_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/all_cells_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_all_cells_ERless_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_all_cells_ERless_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/all_cells_ERless_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_endothelial_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_endothelial_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/endothelial_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_endothelial_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_endothelial_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/endothelial_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_endothelial_stromal_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_endothelial_stromal_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/endothelial_stromal_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_endothelial_stromal_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_endothelial_stromal_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/endothelial_stromal_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_endothelial_tumor_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_endothelial_tumor_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/endothelial_tumor_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_endothelial_tumor_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_endothelial_tumor_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/endothelial_tumor_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_endothelial_ERless_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_immune_endothelial_ERless_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/immune_endothelial_ERless_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_endothelial_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_immune_endothelial_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/immune_endothelial_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_ERless_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_immune_ERless_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/immune_ERless_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_immune_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/immune_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_stromal_ERless_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_immune_stromal_ERless_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/immune_stromal_ERless_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_stromal_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_immune_stromal_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/immune_stromal_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_tumor_ERless_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_immune_tumor_ERless_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/immune_tumor_ERless_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_immune_tumor_radius(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_immune_tumor_radius"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/immune_tumor_radius/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_stromal_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_stromal_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/stromal_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_stromal_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_stromal_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/stromal_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_stromal_tumor_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_stromal_tumor_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/stromal_tumor_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_stromal_tumor_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_stromal_tumor_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/stromal_tumor_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_tumor_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_tumor_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/tumor_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_tumor_ERless_contact(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_tumor_ERless_contact"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/tumor_ERless_contact/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

### Helpers for seed gcl training ###
def get_best_gcl_ER_linear(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_gcl_ER_linear"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/graph_concept_learners/"+"best_gcl_ER_linear/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_gcl_ER_concat(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_gcl_ER_concat"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/graph_concept_learners/"+"best_gcl_ER_concat/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_gcl_ER_transformer(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_gcl_ER_transformer"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/graph_concept_learners/"+"best_gcl_ER_transformer/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_gcl_ERless_linear(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_gcl_ERless_linear"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/graph_concept_learners/"+"best_gcl_ERless_linear/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_gcl_ERless_concat(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_gcl_ERless_concat"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/graph_concept_learners/"+"best_gcl_ERless_concat/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_best_gcl_ERless_transformer(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_gcl_ERless_transformer"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/graph_concept_learners/"+"best_gcl_ERless_transformer/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

### Helper functions for randomly permuted n seed gcl training ###
def get_rnd_gcl_ER_linear(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_gcl_ER_linear"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/graph_concept_learners/randomized_data/"+"best_gcl_ER_linear/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_gcl_ER_concat(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_gcl_ER_concat"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/graph_concept_learners/randomized_data/"+"best_gcl_ER_concat/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

def get_rnd_gcl_ER_transformer(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/best_gcl_ER_transformer"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/graph_concept_learners/randomized_data/"+"best_gcl_ER_transformer/{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])

### Rules fro end 2 end configs ###
def get_end_2_end_conf_mats(wildcards):
    path_to_file = f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/configs/end_2_end_configs"
    CFG_IDS = [os.path.splitext(f)[0] for f in os.listdir(path_to_file) if os.path.splitext(f)[1] == ".yaml"]
    return expand(f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/{split_how}/checkpoints/graph_concept_learners/end_2_end/"+"{cfg_id}/test_conf_mat_from_best_val_{metric_name}.png", cfg_id=CFG_IDS, metric_name=config["follow_this_metrics"])
