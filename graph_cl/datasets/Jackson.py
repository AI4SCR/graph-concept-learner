# %%
import zipfile
from io import BytesIO
import pandas as pd
from pathlib import Path
import shutil
import logging
import json
from ..data_models.PathFactory import PathFactory
from ..data_models.Sample import Sample

logging.basicConfig(level=logging.INFO)


# %%


class Jackson:
    name = "jackson"

    def __init__(self, factory: PathFactory):
        self.factory = factory
        self.sample_path = self.factory.processed_dir / "samples.parquet"
        self.sample_names = []

        # create all parents of the sample paths
        paths = self.factory.get_sample_paths("test_sample", [])
        for k, v in paths.items():
            v.parent.mkdir(exist_ok=True, parents=True)

    def load(self, force=False):
        if not self.sample_path.exists() or force:
            self.extract_metadata()
            self.extract_sample_labels()
            self.extract_observation_labels()
            self.extract_observation_locations()
            self.extract_observation_features()
            self.extract_masks()

            # for each observed sample, we create all potential the paths
            self.sample_names = set(self.sample_names)
            paths = [
                {
                    "sample_name": sample_name,
                    **self.factory.get_sample_paths(sample_name, []),
                }
                for sample_name in self.sample_names
            ]

            paths = pd.DataFrame(paths).set_index("sample_name")
            paths = paths.map(lambda x: str(x) if x.exists() else None)
            paths.to_parquet(self.sample_path)
        else:
            paths = pd.read_parquet(self.sample_path)

        paths.index.name = "name"
        samples = [Sample(**data) for data in paths.reset_index().to_dict("records")]

        self.factory.samples_dir.mkdir(exist_ok=True, parents=True)

        for sample in samples:
            file_path = self.factory.samples_dir / f"{sample.name}.json"
            with file_path.open("w", encoding="utf-8") as file:
                file.write(sample.model_dump_json(indent=4))

        return samples

    def extract_masks(self):
        logging.info("Extracting masks")
        with zipfile.ZipFile(
            self.factory.raw_dir / "OMEandSingleCellMasks.zip", "r"
        ) as zip_ref:
            with zip_ref.open("OMEnMasks/Basel_Zuri_masks.zip") as zip_ext:
                zip_ext_content = zip_ext.read()
                zip_ext_buffer = BytesIO(zip_ext_content)

        with zipfile.ZipFile(zip_ext_buffer) as zip_ref_inner:
            zip_ref_inner.extractall(self.factory.mask_dir)

        return self.rename_masks()

    def rename_masks(self):
        # TODO: this only moves cores which are present in the samples metadata (there are masks that are not listed there)
        df = pd.read_parquet(self.factory.metadata_dir / "samples")

        mask_file_name = [f"{Path(f).stem}_maks.tiff" for f in df["FileName_FullStack"]]
        df = df.assign(mask_file_name=mask_file_name)

        for sample_name, mask_file_name in zip(df.index, df["mask_file_name"]):
            mask_path_old = self.factory.mask_dir / "Basel_Zuri_masks" / mask_file_name
            mask_path_new = self.factory.get_sample_paths(sample_name)["mask_url"]
            mask_path_old.rename(mask_path_new)
            self.sample_names.append(sample_name)

        shutil.rmtree(self.factory.mask_dir / "Basel_Zuri_masks")

    def extract_observation_locations(self):
        logging.info("Extracting observation locations")

        files_to_extract = {
            "bs_sc_locations.csv": "Basel_SC_locations.csv",
            "zh_sc_locations.csv": "Zurich_SC_locations.csv",
        }
        locs = {}
        with zipfile.ZipFile(
            self.factory.raw_dir / "singlecell_locations.zip", "r"
        ) as zip_ref:
            for file_name, filepath in files_to_extract.items():
                if filepath in zip_ref.namelist():
                    buffer = zip_ref.read(filepath)
                    fname = Path(filepath).name
                    locs[fname] = pd.read_csv(BytesIO(buffer))

        df = pd.concat(locs.values())
        for sample_name, grp_data in df.groupby("core"):
            grp_data = grp_data.assign(core=sample_name)
            filepath = self.factory.get_sample_paths(sample_name)["location_url"]
            grp_data.to_parquet(filepath, index=False)
            self.sample_names.append(sample_name)

    def extract_observation_labels(self):
        logging.info("Extracting observation labels")
        files_to_extract = {
            "metacluster_annotations.csv": "Cluster_labels/Metacluster_annotations.csv",
            "zh_pg.csv": "Cluster_labels/PG_basel.csv",
            "bs_pg.csv": "Cluster_labels/PG_zurich.csv",
            "zh_meta_clusters.csv": "Cluster_labels/Zurich_matched_metaclusters.csv",
            "bs_meta_clusters.csv": "Cluster_labels/Basel_metaclusters.csv",
        }
        with zipfile.ZipFile(
            self.factory.raw_dir / "singlecell_cluster_labels.zip", "r"
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
            filepath = self.factory.get_sample_paths(sample_name)["labels_url"]
            grp_data.to_parquet(filepath, index=False)
            self.sample_names.append(sample_name)

    def extract_metadata(self):
        logging.info("Extracting metadata")
        files_to_extract = {
            "staining_panel.csv": "Data_publication/Basel_Zuri_StainingPanel.csv",
        }
        with zipfile.ZipFile(
            self.factory.raw_dir / "SingleCell_and_Metadata.zip", "r"
        ) as zip_ref:
            for file_name, filepath in files_to_extract.items():
                if filepath in zip_ref.namelist():
                    buffer = zip_ref.read(filepath)
                    pd.read_csv(BytesIO(buffer)).to_csv(
                        self.factory.metadata_dir / file_name
                    )

    def extract_sample_labels(self):
        logging.info("Extracting sample labels")
        files_to_extract = {
            "bs_patient_meta_data.csv": "Data_publication/BaselTMA/Basel_PatientMetadata.csv",
            "zh_patient_meta_data.csv": "Data_publication/ZurichTMA/Zuri_PatientMetadata.csv",
        }

        cont = {}
        with zipfile.ZipFile(
            self.factory.raw_dir / "SingleCell_and_Metadata.zip", "r"
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
            labels_path = self.factory.get_sample_paths(sample_name)[
                "sample_labels_url"
            ]
            labels.loc[[sample_name]].to_parquet(labels_path)
            self.sample_names.append(sample_name)

        for sample_name in labels.index:
            filepath = self.factory.get_sample_paths(sample_name)["metadata_url"]
            metadata.loc[[sample_name]].to_parquet(filepath)
            self.sample_names.append(sample_name)

    def extract_observation_features(self):
        logging.info("Extracting observation features")

        files_to_extract = {
            "bs_sc_dat.csv": "Data_publication/BaselTMA/SC_dat.csv",
            "zh_sc_dat.csv": "Data_publication/ZurichTMA/SC_dat.csv",
            "staining_panel": "Data_publication/Basel_Zuri_StainingPanel.csv",
        }
        cont = {}
        with zipfile.ZipFile(
            self.factory.raw_dir / "SingleCell_and_Metadata.zip", "r"
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

            expr_path = self.factory.get_sample_paths(sample_name)["expression_url"]
            x_expr.reset_index(drop=False).to_parquet(expr_path)

            spatial_path = self.factory.get_sample_paths(sample_name)["spatial_url"]
            x_spatial.reset_index(drop=False).to_parquet(spatial_path)

            self.sample_names.append(sample_name)
