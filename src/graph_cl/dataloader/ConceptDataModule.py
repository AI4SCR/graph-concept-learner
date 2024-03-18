import lightning as L
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import torch

from ai4bmr_core.log.log import logger
from ..data_models.Data import DataConfig

from ..preprocessing.normalize import Normalizer
from ..data_models.Sample import Sample
from ..data_models.Experiment import GCLExperiment


class ConceptDataset(Dataset):
    def __init__(self, data: list[Data] | list[dict[str, Data]]):
        self.data = data
        super().__init__()

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


class ConceptDataModule(L.LightningDataModule):
    def __init__(
        self,
        splits: dict[str, list[Sample]],
        model_name: str,
        concepts: str | list[str],
        config: DataConfig,
        factory: GCLExperiment,
        batch_size: int = 8,
        shuffle: bool = True,
        force_attr_computation: bool = False,
    ):
        # TODO: where should we seed?
        from torch_geometric import seed_everything

        seed_everything(config.seed)

        super().__init__()

        # init params
        self.splits = splits
        self.model_name = model_name
        self.concepts = concepts
        if self.model_name == "gnn":
            assert isinstance(self.concepts, str)
        elif self.model_name == "gcl":
            assert isinstance(self.concepts, list)
        else:
            raise ValueError(f"model_name={self.model_name} not supported")

        self.config = config
        self.factory = factory
        self.save_samples_dir = self.factory._experiment_samples_dir

        # NOTE: we cannot use a simpler solution and just overwrite the current datasets for each
        #   model and concept, because if we train in parallel and share the storage we might load the wrong
        #   dataset.
        concept_name = self.concepts if self.model_name == "gnn" else ""
        self.save_dataset_dir = self.factory.get_experiment_dataset_path(
            model_name=self.model_name, concept_name=concept_name
        )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.force_attr_computation = force_attr_computation

        # computed field
        self.normalize = Normalizer(**self.config.normalize.kwargs)
        self.num_features = self.splits["fit"][0].expression.shape[1]

        # datasets
        self.ds_fit = None
        self.ds_val = None
        self.ds_test = None

    def prepare_data(self):
        from ..preprocessing.attribute import collect_features, collect_sample_features

        feats = collect_features(
            samples=self.splits["fit"], feature_dicts=self.config.features
        )

        self.normalize.fit(feats)
        self.num_features = feats.shape[1]

        for stage in ["fit", "val", "test", "predict"]:
            if stage not in self.splits:
                continue

            samples = self.splits[stage]
            ds = []
            for s in samples:
                assert (
                    s.stage == stage
                )  # enforce that samples are labelled with the correct stage they belong to
                if s.attributes is None or self.force_attr_computation:
                    sample_feats = collect_sample_features(
                        s, feature_dicts=self.config.features
                    )
                    attributes = self.normalize.transform(sample_feats)
                    attributes_url = (
                        self.factory.get_attribute_dir(stage) / f"{s.name}.parquet"
                    )
                    s.attributes_url = attributes_url
                    attributes.to_parquet(s.attributes_url)
                    s.model_dump_to_json(self.save_samples_dir / f"{s.name}.json")

                if isinstance(self.concepts, str):
                    cg = s.attributed_graph(self.concepts)
                else:
                    cg = {
                        concept_name: s.attributed_graph(concept_name)
                        for concept_name in self.concepts
                    }
                ds.append(cg)

            torch.save(ds, self.save_dataset_dir / f"{stage}.pt")
            logger.info(f"saved `{stage}.pt` dataset to {self.save_dataset_dir}")

    def setup(self, stage: str):
        # note: it seems to be necessary to setup all dataloaders or at least the fit, val loaders.
        #   the val_dataloader is called after just running the setup with stage='fit'. Unclear why.
        #   -> probably because if we `fit` we also have to validate, thus lightning forces you to setup both.
        #   I would prefer a syntax like this, but the call order is setup('fit') -> val_dataloader('val')
        # if stage == "fit":
        #     self.ds_fit = torch.load(self.save_dataset_dir / f"fit.pt")

        if "fit" in self.splits:
            path_save = self.save_dataset_dir / f"fit.pt"
            self.ds_fit = torch.load(path_save)
        if "val" in self.splits or "validate" in self.splits:
            path_save = self.save_dataset_dir / f"val.pt"
            self.ds_val = torch.load(path_save)
        if "test" in self.splits:
            path_save = self.save_dataset_dir / f"test.pt"
            self.ds_val = torch.load(path_save)

    def train_dataloader(self):
        return DataLoader(self.ds_fit, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass
