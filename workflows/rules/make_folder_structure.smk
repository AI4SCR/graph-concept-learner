rule make_folder_structure:
    input:
        "./config/main_config.yaml"
    resources:
        cores=1,
        mem="1G",
        queue="x86_1h",
    shell:
        "scripts/make_folder_structure.py {input}"
