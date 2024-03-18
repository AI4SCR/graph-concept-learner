import pickle
from pathlib import Path


class JSONMixIn:
    @classmethod
    def model_validate_from_json(cls, path):
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def model_dump_to_json(self, path: Path | str):
        with open(path, "w") as f:
            f.write(self.model_dump_json())


class YAMLMixIN:
    # TODO: add typing
    @classmethod
    def model_validate_from_json(cls, path, **kwargs):
        """Load data model from a yaml file. If the file contains a list of items, a list of data models is returned.

        Args:
            path: path to the yaml file
            **kwargs: arguments that override the values loaded from the yaml file. Useful for values that are only
                known at runtime. This allows to define the value in the yaml file as null and override it at runtime.

        Returns:
            cls instance or list of cls instances
        """
        import yaml

        with open(path) as f:
            items = yaml.safe_load(f)
            if isinstance(items, list):
                return [cls(**{**i, **kwargs}) for i in items]
            else:
                return cls(**{**items, **kwargs})


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
