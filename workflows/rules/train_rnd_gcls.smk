rule train_rnd_gcls:
    input:
        get_rnd_gcl_ER_linear,
        get_rnd_gcl_ER_concat,
        get_rnd_gcl_ER_transformer,
