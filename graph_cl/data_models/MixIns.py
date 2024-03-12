import pickle
from pathlib import Path


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
        self, path: Path | str, overwrite: bool = False, check_integrity: bool = True
    ):
        if check_integrity:
            assert self.check_integrity()
        if overwrite is False:
            assert Path(path).exists() is False
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
