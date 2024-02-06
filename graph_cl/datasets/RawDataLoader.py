# %%
import zipfile
from io import BytesIO
from skimage import io
import pandas as pd
import numpy as np
from spatialOmics import SpatialOmics
from pathlib import Path
import pickle


# %%


class RawDataLoader:
    def __init__(self, configuration):
        self.configuration = configuration

    def extract_masks(self):
        with zipfile.ZipFile(
            self.configuration.data.raw / "OMEandSingleCellMasks.zip", "r"
        ) as zip_ref:
            with zip_ref.open("OMEnMasks/Basel_Zuri_masks.zip") as file:
                inner_zip_contents = file.read()

        # Now, treat inner_zip_contents as a BytesIO object
        inner_zip_buffer = BytesIO(inner_zip_contents)

        with zipfile.ZipFile(inner_zip_buffer) as inner_zip_ref:
            # read all tiff files from the inner zip file
            masks = {}
            for f in inner_zip_ref.namelist():
                if f.endswith(".tiff"):
                    fname = Path(f).name
                    buffer = inner_zip_ref.read(f)
                    masks[fname] = io.imread(BytesIO(buffer))

        return masks

    def extract_single_cell_locations(self):
        with zipfile.ZipFile(
            self.configuration.data.raw / "singlecell_locations.zip", "r"
        ) as zip_ref:
            locs = {}
            for f in zip_ref.namelist():
                if f.endswith(".csv"):
                    fname = Path(f).name
                    buffer = zip_ref.read(f)
                    locs[fname] = pd.read_csv(BytesIO(buffer))
        return locs

    def extract_single_cell_cluster_labels(self):
        with zipfile.ZipFile(
            self.configuration.data.raw / "singlecell_cluster_labels.zip", "r"
        ) as zip_ref:
            labels = {}
            for f in zip_ref.namelist():
                if f.endswith(".csv"):
                    fname = Path(f).name
                    buffer = zip_ref.read(f)
                    labels[fname] = pd.read_csv(
                        BytesIO(buffer),
                        sep=";" if fname == "Metacluster_annotations.csv" else ",",
                    )
        return labels

    def extract_meta_data(self):
        with zipfile.ZipFile(
            self.configuration.data.raw / "SingleCell_and_Metadata.zip", "r"
        ) as zip_ref:
            meta = {}
            for f in zip_ref.namelist():
                if f.endswith(".csv"):
                    fname = Path(f).name
                    fname = (
                        f"ZH_{fname}"
                        if "/ZurichTMA/" in f
                        else f"BS_{fname}"
                        if "/BaselTMA/" in f
                        else fname
                    )
                    buffer = zip_ref.read(f)
                    meta[fname] = pd.read_csv(BytesIO(buffer))
        return meta

    @staticmethod
    def gather_spl_metadata(df_zh, df_bs):
        # Add cohort column
        df_zh["cohort"] = "zurich"
        df_bs["cohort"] = "basel"

        # Get colum interections
        z = set(df_zh.columns.values)
        b = set(df_bs.columns.values)
        intx = z.intersection(b)

        # Concatenate
        md = pd.concat([df_zh[list(intx)], df_bs[list(intx)]])
        return md

    @staticmethod
    def gather_ct_data(labels, locs):

        # NOTE: ['core', 'CellId', 'PhenoGraph', 'id']
        pg_zh = labels["PG_zurich.csv"]
        pg_bs = labels["PG_basel.csv"]
        zh_meta_clusters = labels["Zurich_matched_metaclusters.csv"]
        bs_meta_clusters = labels["Basel_metaclusters.csv"]
        meta_cluster_anno = labels["Metacluster_annotations.csv"]
        meta_cluster_anno.columns = meta_cluster_anno.columns.str.strip()

        loc_zh = locs["Zurich_SC_locations.csv"]
        loc_bs = locs["Basel_SC_locations.csv"]

        zh = pg_zh.merge(zh_meta_clusters, on="id", how="outer").merge(
            loc_zh, on="id", how="outer"
        )
        bs = pg_bs.merge(bs_meta_clusters, on="id", how="outer").merge(
            loc_bs, on="id", how="outer"
        )

        # TODO: can we do that?
        zh = zh[zh.cluster.notna()]
        assert (zh.core_x == zh.core_y).mean() == 1
        zh = zh.assign(core=zh.core_x)
        zh = zh.drop(columns=["core_x", "core_y"])

        bs = bs[bs.cluster.notna()]
        assert (bs.core_x == bs.core_y).mean() == 1
        bs = bs.assign(core=bs.core_x, PhenoGraph=bs.PhenoGraphBasel)
        bs = bs.drop(columns=["core_x", "core_y", "PhenoGraphBasel"])

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

        return df

    @staticmethod
    def split_scc_data(meta):
        sc_bs = meta["BS_SC_dat.csv"]
        sc_zh = meta["ZH_SC_dat.csv"]
        X = pd.concat([sc_bs, sc_zh])
        return X

    def __call__(self, *args, **kwargs):
        masks = self.extract_masks()
        locs = self.extract_single_cell_locations()
        # NOTE: 'PG_zurich.csv' is in both labels and meta, the df is identical
        labels = self.extract_single_cell_cluster_labels()
        meta = self.extract_meta_data()

        spl_df = self.gather_spl_metadata(
            meta["ZH_Zuri_PatientMetadata.csv"], meta["BS_Basel_PatientMetadata.csv"]
        )

        so = SpatialOmics()

        so.spl = spl_df
        uns_df = meta["Basel_Zuri_StainingPanel.csv"]
        uns_df = uns_df.dropna(axis=0, how="all")
        so.uns["staining_panel"] = uns_df

        # Get list of all samples.
        # NOTE: after my migration, this are integers 0-N and 0-M with N,M being the number of samples in each cohort.
        #   This also means that there are REPEATING sample numbers!
        spls = spl_df.index.values
        cores = spl_df.core.values

        obs = self.gather_ct_data(labels, locs)
        X = self.split_scc_data(meta)

        # Make map for numerica cell_type class
        keys = ["Stroma", "Immune", "Vessel", "Tumor"]
        numeric_labels = list(range(0, len(keys)))
        map_to_numeric = dict(zip(keys, numeric_labels))

        colnames_X_wide = X.channel.values

        # Get metal tag labels
        unique_metal_tags = np.unique(uns_df["Metal Tag"].values)

        # Init dictionary
        map_to_metaltag = {}

        # Map matal tag label to column name
        for string in colnames_X_wide:
            for substring in unique_metal_tags:
                if substring in string:
                    map_to_metaltag[string] = substring

        for grp_name, grp_X in X.groupby("core"):
            # Load obs df
            grp_obs = obs[obs.core == grp_name]
            grp_obs = grp_obs.assign(class_id=["Class"].map(map_to_numeric))
            grp_obs.index.names = ["cell_id"]
            so.obs[grp_name] = grp_obs[
                [
                    "id",
                    "cluster",
                    "Cell_type",
                    "Class",
                    "class_id",
                    "PhenoGraph",
                    "Location_Center_X",
                    "Location_Center_Y",
                ]
            ]

            grp_X = grp_X.pivot(index="CellId", columns="channel", values="mc_counts")
            grp_X.index.names = ["cell_id"]
            so.X[grp_name] = grp_X

            # Join morphological features into obs
            so.obs[grp_name] = so.obs[grp_name].join(
                so.X[grp_name][
                    [
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
                ],
                how="left",
            )

            # Colnames in lowercase
            so.obs[grp_name].rename(str.lower, axis="columns", inplace=True)

            # Rename class as cell_class
            so.obs[grp_name].rename(columns={"class": "cell_class"}, inplace=True)

            # Reneame columns with metal tag
            so.X[grp_name] = so.X[grp_name].rename(columns=map_to_metaltag)

            # Get metal tags present in sample
            metaltas_in_sampl = so.X[grp_name].columns.intersection(unique_metal_tags)

            # Drop other columns
            so.X[grp_name] = so.X[grp_name][metaltas_in_sampl]

            # Load mask
            mask_fname = list(filter(lambda x: spl in x, masks.keys()))[0]
            so.masks[grp_name]["cellmasks"] = masks[mask_fname]

        # Drop metal tags only present in either of the cohorts
        metal_tags_in_all_spls = set(unique_metal_tags)
        for spl in cores:
            metal_tags_in_all_spls = metal_tags_in_all_spls.intersection(
                so.X[spl].columns
            )

        for spl in cores:
            # Drop columns
            so.X[spl] = so.X[spl][list(metal_tags_in_all_spls)]

        ### Write to file ###
        with open(self.configuration.data.intermediate, "wb") as f:
            pickle.dump(so, f)

        return so
