def download_dataset(dataset_name):
    NotImplementedError(f"Dataset {dataset_name} not implemented")


def process_dataset(dataset_name):
    from ...data_models.DatasetPathFactory import DatasetPathFactory
    from ...datasets.Jackson import Jackson

    factory = DatasetPathFactory(dataset_name=dataset_name)

    if dataset_name.lower() == "jackson":
        ds = Jackson(factory=factory)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    ds.process()
