from .Jackson import Jackson


def get_dataset_by_name(dataset_name: str):
    dataset_name = dataset_name.strip().lower()
    if dataset_name == "jackson":
        return Jackson()
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
