# Make so object from the preporcessed files
rule make_so:
    input:
        f"{root}/raw_data/unzipped/Data_publication/Basel_Zuri_StainingPanel.csv",
        f"{root}/intermediate_data/spl_meta_data.csv",
        f"{root}/intermediate_data/sct_data",
        f"{root}/intermediate_data/scc_data",
        f"{root}/raw_data/unzipped/OMEnMasks/Basel_Zuri_masks"
    output:
        f"{root}/intermediate_data/so.pkl"
    resources:
        cores=2,
        mem="16G",
        queue="x86_1h",
    log:
        f"{root}/logs/make_so/make_so"
    shell:
        "0_make_so_jakson/scripts/make_so.py {input} {output}"
