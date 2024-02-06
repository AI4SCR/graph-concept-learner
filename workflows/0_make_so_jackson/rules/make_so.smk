# Make so object from the preporcessed files
rule make_so:
    input:
        f"{root}/raw_data/unzipped/Data_publication/Basel_Zuri_StainingPanel.csv",  # uns_path
        f"{root}/intermediate_data/spl_meta_data.csv",  # spl_path
        f"{root}/intermediate_data/sct_data",  # obs_dir
        f"{root}/intermediate_data/scc_data",  # X_dir
        f"{root}/raw_data/unzipped/OMEnMasks/Basel_Zuri_masks"  # masks_dir
    output:
        f"{root}/intermediate_data/so.pkl"
    resources:
        cores=2,
        mem="16G",
        queue="x86_1h",
    log:
        f"{root}/logs/make_so/make_so"
    shell:
        "0_make_so_jackson/scripts/make_so.py {input} {output}"
