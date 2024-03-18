def download(dataset_name):
    NotImplementedError(f"Dataset {dataset_name} not implemented")


def process(dataset_name):
    from ...datasets.Jackson import Jackson

    if dataset_name.lower() == "jackson":
        ds = Jackson()
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    ds.process()
