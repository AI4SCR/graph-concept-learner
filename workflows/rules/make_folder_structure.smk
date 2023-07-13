rule make_folder_structure:
    input:
        main_cfg = "./config/main_config.yaml",
        c1 = "./default_configs/concept_1_radius.yaml",
        c2 = "./default_configs/concept_2_knn.yaml",
        c3 = "./default_configs/concept_3_contact.yaml",
        pretrain = "./default_configs/pretrain_models_base_config.yaml",
        train = "./default_configs/train_models_base_config.yaml",
        attributes = "./default_configs/all_X_cols.yaml",
    resources:
        cores=1,
        mem="1G",
        queue="x86_1h",
    params:
        path_to_cfgs=f"{root}/prediction_tasks/{prediction_target}/{normalized_with}/configs"
    shell:
        "scripts/make_folder_structure.py {input.main_cfg} && "
        "cp -t {params.path_to_cfgs}/dataset_configs/ {input.c1} {input.c2} {input.c3} && "
        "cp -t {params.path_to_cfgs}/base_configs/ {input.pretrain} {input.train} && "
        "cp {input.attributes} {params.path_to_cfgs}/attribute_configs/"
