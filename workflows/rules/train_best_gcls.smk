rule train_best_gcls:
    input:
        get_best_gcl_ER_linear,
        get_best_gcl_ER_concat,
        get_best_gcl_ER_transformer,
        get_best_gcl_ERless_linear,
        get_best_gcl_ERless_concat,
        get_best_gcl_ERless_transformer,
