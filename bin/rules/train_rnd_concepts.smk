rule train_rnd_concepts:
    input:
        get_rnd_all_cells_contact,
        get_rnd_all_cells_radius,
        get_rnd_endothelial_contact,
        get_rnd_endothelial_stromal_contact,
        get_rnd_endothelial_tumor_contact,
        get_rnd_immune_endothelial_radius,
        get_rnd_immune_radius,
        get_rnd_immune_stromal_radius,
        get_rnd_immune_tumor_radius,
        get_rnd_stromal_contact,
        get_rnd_stromal_tumor_contact,
        get_rnd_tumor_contact,
