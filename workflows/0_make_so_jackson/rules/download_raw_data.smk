# Download Jakson raw data
rule download_raw_data:
    input:
        config["raw_data_dir"] + "/download_info/download_urls.txt"
    output:
        directory(config["raw_data_dir"] + "/zipped"),
    log:
        config["raw_data_dir"] + "/download_info/download_logs"
    shell:
        "0_make_so_jackson/scripts/download_raw_data.sh {input} {config[raw_data_dir]}"
