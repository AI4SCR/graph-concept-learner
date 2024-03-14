import pickle
from pathlib import Path


class ModelIOMixIn:
    @classmethod
    def model_validate_from_file(cls, path):
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def model_dump_to_file(self, path: Path | str):
        with open(path, "w") as f:
            f.write(self.model_dump_json())


class FromYamlMixIn:
    # TODO: add typing
    @classmethod
    def from_yaml(cls, path):
        import yaml

        with open(path) as f:
            items = yaml.safe_load(f)
            if isinstance(items, list):
                return [cls(**i) for i in items]
            else:
                return cls(**items)


class PickleMixIn:
    def to_pickle(
        self, path: Path | str, exists: str = "skip", check_integrity: bool = True
    ):
        if check_integrity and hasattr(self, "check_integrity"):
            assert self.check_integrity()

        if Path(path).exists():
            if exists == "skip":
                return
            elif exists == "overwrite":
                pass
            elif exists == "raise":
                raise FileExistsError(f"{path} already exists")

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
