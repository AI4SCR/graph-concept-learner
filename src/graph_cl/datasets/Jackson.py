# %%
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from ai4bmr_core.log.log import logger

from .GCLDataset import GCLDataset
from ..data_models.Sample import Sample


# %%


class Jackson(GCLDataset):
    # dataset fields, required
    _id: str = "jackson-v1.0.0"
    _name: str = "jackson"
    _data: list[Sample] = []
    # setting the _urls field will enable the download method
    _urls = {
        "OMEandSingleCellMasks.zip": "https://zenodo.org/records/4607374/files/OMEandSingleCellMasks.zip?download=1",
        "singlecell_locations.zip": "https://zenodo.org/records/4607374/files/singlecell_locations.zip?download=1",
        "singlecell_cluster_labels.zip": "https://zenodo.org/records/4607374/files/singlecell_cluster_labels.zip?download=1",
        "SingleCell_and_Metadata.zip": "https://zenodo.org/records/4607374/files/SingleCell_and_Metadata.zip?download=1",
    }

    # user fields, optional, additional
    sample_names: list[str] | set[str] = []

    @property
    def processed_files(self) -> list[Path]:
        # adding this property enables caching
        if not self.info_path.exists():
            return []
        paths = pd.read_parquet(self.info_path)
        return paths.name.map(lambda x: self.samples_dir / f"{x}.json").tolist()

    def load_cache(self) -> list[Sample]:
        return [
            Sample.model_validate_from_json(p) for p in self.samples_dir.glob("*.json")
        ]

    def save_cache(self, samples: list[Sample]) -> None:
        for sample in samples:
            file_path = self.samples_dir / f"{sample.name}.json"
            with file_path.open("w", encoding="utf-8") as file:
                file.write(sample.model_dump_json(indent=4))

    @staticmethod
    def harmonize_indices(df_samples: pd.DataFrame):
        logger.info("Harmonizing indices")
        from graph_cl.preprocessing.harmonize import harmonize_index

        for sample in df_samples.index:
            harmonize_index(
                mask_path=df_samples.loc[sample].mask_url,
                expr_path=df_samples.loc[sample].expression_url,
                labels_path=df_samples.loc[sample].labels_url,
                loc_path=df_samples.loc[sample].location_url,
                spat_path=df_samples.loc[sample].spatial_url,
            )

    def load(self):
        self.create_folder_hierarchy()

        self.extract_metadata()
        self.extract_sample_labels()
        self.extract_observation_labels()
        self.extract_observation_locations()
        self.extract_observation_features()
        self.extract_masks()

        self.sample_names = set(self.sample_names)
        paths = [
            {
                "name": sample_name,
                **self.get_sample_paths(sample_name),
            }
            for sample_name in self.sample_names
        ]

        paths = pd.DataFrame(paths).set_index("name")
        paths = paths.map(lambda x: str(x) if x.exists() else None)

        # TODO: we can only harmonize indices if we have all the data
        cols = [
            "mask_url",
            "expression_url",
            "labels_url",
            "location_url",
            "spatial_url",
        ]
        self.harmonize_indices(paths[cols].dropna())

        paths = paths.reset_index()
        paths.to_parquet(self.info_path)

        samples = [Sample(**data) for data in paths.to_dict("records")]

        return samples

    def extract_masks(self):
        logger.info("Extracting masks")
        with zipfile.ZipFile(
            self.raw_dir / "OMEandSingleCellMasks.zip", "r"
        ) as zip_ref:
            with zip_ref.open("OMEnMasks/Basel_Zuri_masks.zip") as zip_ext:
                zip_ext_content = zip_ext.read()
                zip_ext_buffer = BytesIO(zip_ext_content)

        with zipfile.ZipFile(zip_ext_buffer) as zip_ref_inner:
            zip_ref_inner.extractall(self.mask_dir)

        return self.rename_masks()

    def rename_masks(self):
        # TODO: this only moves cores which are present in the samples metadata (there are masks that are not listed there)
        df = pd.read_parquet(self.metadata_dir / "samples")

        mask_file_name = [f"{Path(f).stem}_maks.tiff" for f in df["FileName_FullStack"]]
        df = df.assign(mask_file_name=mask_file_name)

        for sample_name, mask_file_name in zip(df.index, df["mask_file_name"]):
            mask_path_old = self.mask_dir / "Basel_Zuri_masks" / mask_file_name
            mask_path_new = self.get_sample_paths(sample_name)["mask_url"]
            mask_path_old.rename(mask_path_new)
            self.sample_names.append(sample_name)

        shutil.rmtree(self.mask_dir / "Basel_Zuri_masks")

    def extract_observation_locations(self):
        logger.info("Extracting observation locations")

        files_to_extract = {
            "bs_sc_locations.csv": "Basel_SC_locations.csv",
            "zh_sc_locations.csv": "Zurich_SC_locations.csv",
        }
        locs = {}
        with zipfile.ZipFile(self.raw_dir / "singlecell_locations.zip", "r") as zip_ref:
            for file_name, filepath in files_to_extract.items():
                if filepath in zip_ref.namelist():
                    buffer = zip_ref.read(filepath)
                    fname = Path(filepath).name
                    locs[fname] = pd.read_csv(BytesIO(buffer))

        df = pd.concat(locs.values())
        for sample_name, grp_data in df.groupby("core"):
            grp_data = grp_data.assign(core=sample_name)
            filepath = self.get_sample_paths(sample_name)["location_url"]
            grp_data.to_parquet(filepath, index=False)
            self.sample_names.append(sample_name)

    def extract_observation_labels(self):
        logger.info("Extracting observation labels")
        files_to_extract = {
            "metacluster_annotations.csv": "Cluster_labels/Metacluster_annotations.csv",
            "zh_pg.csv": "Cluster_labels/PG_basel.csv",
            "bs_pg.csv": "Cluster_labels/PG_zurich.csv",
            "zh_meta_clusters.csv": "Cluster_labels/Zurich_matched_metaclusters.csv",
            "bs_meta_clusters.csv": "Cluster_labels/Basel_metaclusters.csv",
        }
        with zipfile.ZipFile(
            self.raw_dir / "singlecell_cluster_labels.zip", "r"
        ) as zip_ref:
            labels = {}
            for file_name, filepath in files_to_extract.items():
                if filepath in zip_ref.namelist():
                    fname = Path(filepath).name
                    buffer = zip_ref.read(filepath)
                    labels[fname] = pd.read_csv(
                        BytesIO(buffer),
                        sep=";" if fname == "Metacluster_annotations.csv" else ",",
                    )

        # NOTE: ['core', 'CellId', 'PhenoGraph', 'id']
        pg_zh = labels["PG_zurich.csv"]
        pg_bs = labels["PG_basel.csv"]
        zh_meta_clusters = labels["Zurich_matched_metaclusters.csv"]
        bs_meta_clusters = labels["Basel_metaclusters.csv"]
        meta_cluster_anno = labels["Metacluster_annotations.csv"]
        meta_cluster_anno.columns = meta_cluster_anno.columns.str.strip()

        zh = pg_zh.merge(zh_meta_clusters, on="id", how="outer")
        bs = pg_bs.merge(bs_meta_clusters, on="id", how="outer")

        # TODO: can we just drop?
        zh = zh[zh.cluster.notna()]
        bs = bs[bs.cluster.notna()]
        bs = bs.assign(PhenoGraph=bs.PhenoGraphBasel)
        bs = bs.drop(columns=["PhenoGraphBasel"])

        assert set(zh.id).intersection(set(bs.id)) == set()

        df = pd.concat([zh, bs])
        assert df.isna().sum().sum() == 0

        df = df.assign(
            CellId=df.CellId.astype(int),
            PhenoGraph=df.PhenoGraph.astype(int),
            cluster=df.cluster.astype(int),
        )

        df = df.merge(
            meta_cluster_anno, left_on="cluster", right_on="Metacluster", how="left"
        )
        # Make map for numerica cell_type class
        df = df.assign(cell_class_id=df.groupby("Class").ngroup())
        df = df.rename(columns={"Cell type": "cell_type", "Class": "cell_class"})
        for sample_name, grp_data in df.groupby("core"):
            grp_data = grp_data.assign(core=sample_name)
            filepath = self.get_sample_paths(sample_name)["labels_url"]
            grp_data.to_parquet(filepath, index=False)
            self.sample_names.append(sample_name)

    def extract_metadata(self):
        logger.info("Extracting metadata")
        files_to_extract = {
            "staining_panel.csv": "Data_publication/Basel_Zuri_StainingPanel.csv",
        }
        with zipfile.ZipFile(
            self.raw_dir / "SingleCell_and_Metadata.zip", "r"
        ) as zip_ref:
            for file_name, filepath in files_to_extract.items():
                if filepath in zip_ref.namelist():
                    buffer = zip_ref.read(filepath)
                    pd.read_csv(BytesIO(buffer)).to_csv(self.metadata_dir / file_name)

    def extract_sample_labels(self):
        logger.info("Extracting sample labels")
        files_to_extract = {
            "bs_patient_meta_data.csv": "Data_publication/BaselTMA/Basel_PatientMetadata.csv",
            "zh_patient_meta_data.csv": "Data_publication/ZurichTMA/Zuri_PatientMetadata.csv",
        }

        cont = {}
        with zipfile.ZipFile(
            self.raw_dir / "SingleCell_and_Metadata.zip", "r"
        ) as zip_ref:
            for file_name, filepath in files_to_extract.items():
                if filepath in zip_ref.namelist():
                    buffer = zip_ref.read(filepath)
                    cont[file_name] = pd.read_csv(BytesIO(buffer))

        zh = cont["zh_patient_meta_data.csv"]
        bs = cont["bs_patient_meta_data.csv"]

        # Add cohort column
        zh["cohort"] = "zurich"
        bs["cohort"] = "basel"

        # TODO: this is too simple atm, check that columns are not just named differently for ER, HER2, ...
        z = set(zh.columns.values)
        b = set(bs.columns.values)
        intx = z.intersection(b)

        # TODO: only select labels and not metadata too
        metadata_cols = {
            "FileName_FullStack",
            "Height_FullStack",
            "TMABlocklabel",
            "TMALocation",
            "TMAxlocation",
            "Tag",
            "Width_FullStack",
            "Yearofsamplecollection",
            "yLocation",
            "AllSamplesSVSp4.Array",
            "Comment",
        }

        df = pd.concat([zh[list(intx)], bs[list(intx)]])
        assert (df.core.value_counts() == 1).all()

        metadata = df[["core"] + list(metadata_cols)]
        # NOTE: hack to avoid NaNs in the parquet file and thus be of undefined type
        # TODO: fix dtypes
        metadata = metadata.astype(str)
        metadata = metadata.set_index("core")

        labels = df[list(intx - metadata_cols)]
        labels = labels.astype(str).set_index("core")
        for sample_name in labels.index:
            labels_path = self.get_sample_paths(sample_name)["labels_sample_url"]
            labels.loc[[sample_name]].to_parquet(labels_path)
            self.sample_names.append(sample_name)

        for sample_name in labels.index:
            filepath = self.get_sample_paths(sample_name)["metadata_url"]
            metadata.loc[[sample_name]].to_parquet(filepath)
            self.sample_names.append(sample_name)

    def extract_observation_features(self):
        logger.info("Extracting observation features")

        files_to_extract = {
            "bs_sc_dat.csv": "Data_publication/BaselTMA/SC_dat.csv",
            "zh_sc_dat.csv": "Data_publication/ZurichTMA/SC_dat.csv",
            "staining_panel": "Data_publication/Basel_Zuri_StainingPanel.csv",
        }
        cont = {}
        with zipfile.ZipFile(
            self.raw_dir / "SingleCell_and_Metadata.zip", "r"
        ) as zip_ref:
            for file_name, filepath in files_to_extract.items():
                if filepath in zip_ref.namelist():
                    buffer = zip_ref.read(filepath)
                    cont[file_name] = pd.read_csv(BytesIO(buffer))

        # we discard the first 9 rows of the staining panel because they are not used
        panel = cont["staining_panel"][9:]

        b, z = set(cont["bs_sc_dat.csv"].columns), set(cont["zh_sc_dat.csv"].columns)

        cb = set(cont["bs_sc_dat.csv"]["channel"])
        cz = set(cont["zh_sc_dat.csv"]["channel"])
        channels_not_in_both_cohorts = (cb - cz).union(cz - cb)  # not empty
        channels_in_both_cohorts = cb.intersection(cz)

        X = pd.concat([cont["bs_sc_dat.csv"], cont["zh_sc_dat.csv"]])

        spatial_feat = [
            "Area",
            "Eccentricity",
            "EulerNumber",
            "Extent",
            "MajorAxisLength",
            "MinorAxisLength",
            "Number_Neighbors",
            "Orientation",
            "Percent_Touching",
            "Perimeter",
            "Solidity",
        ]

        uniq_expr_feat = channels_in_both_cohorts - set(spatial_feat)

        # we try to map the channel names to the targets in the panel
        # NOTE: some metal tags appear twice -> panel["Metal Tag"].value_counts()
        c2t = {}
        for channel in uniq_expr_feat:
            c2t[channel] = list(filter(lambda x: x in channel, panel["Metal Tag"]))

        valid_channels = list(filter(lambda x: len(c2t[x]) == 1, c2t))
        channel_names = list(map(lambda x: c2t[x][0], valid_channels))
        map_channels = dict(zip(valid_channels, channel_names))

        for sample_name, grp_x in X.groupby("core"):
            grp_x = grp_x.pivot(index="CellId", columns="channel", values="mc_counts")
            grp_x = (
                grp_x.assign(core=sample_name)
                .reset_index(drop=False)
                .set_index(["CellId", "core"])
            )

            x_spatial = grp_x[spatial_feat]
            x_expr = grp_x.drop(columns=spatial_feat)
            x_expr = x_expr[valid_channels].rename(columns=map_channels)

            expr_path = self.get_sample_paths(sample_name)["expression_url"]
            x_expr.reset_index(drop=False).to_parquet(expr_path)

            spatial_path = self.get_sample_paths(sample_name)["spatial_url"]
            x_spatial.reset_index(drop=False).to_parquet(spatial_path)

            self.sample_names.append(sample_name)
