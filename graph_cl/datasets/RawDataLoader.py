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
    def __init__(self, raw_dir: Path, output: Path):
        self.raw = raw_dir
        self.output = output

    def extract_masks(self):
        with zipfile.ZipFile(self.raw / "OMEandSingleCellMasks.zip", "r") as zip_ref:
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
        with zipfile.ZipFile(self.raw / "singlecell_locations.zip", "r") as zip_ref:
            locs = {}
            for f in zip_ref.namelist():
                if f.endswith(".csv"):
                    fname = Path(f).name
                    buffer = zip_ref.read(f)
                    locs[fname] = pd.read_csv(BytesIO(buffer))
        return locs

    def extract_single_cell_cluster_labels(self):
        with zipfile.ZipFile(
            self.raw / "singlecell_cluster_labels.zip", "r"
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
        with zipfile.ZipFile(self.raw / "SingleCell_and_Metadata.zip", "r") as zip_ref:
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
        md = md.set_index("core")
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
        df = df.assign(Cell_type=df["Cell type"]).drop(columns=["Cell type"])

        return df

    @staticmethod
    def split_scc_data(meta):
        sc_bs = meta["BS_SC_dat.csv"]
        sc_zh = meta["ZH_SC_dat.csv"]
        X = pd.concat([sc_bs, sc_zh])
        return X

    def create_so(self, *args, **kwargs):
        if self.output.exists():
            return
        masks = self.extract_masks()
        locs = self.extract_single_cell_locations()
        # NOTE: 'PG_zurich.csv' is in both labels and meta, the df is identical
        labels = self.extract_single_cell_cluster_labels()
        meta = self.extract_meta_data()
        spl_df = self.gather_spl_metadata(
            meta["ZH_Zuri_PatientMetadata.csv"], meta["BS_Basel_PatientMetadata.csv"]
        )

        so = SpatialOmics()

        # attr: uns
        uns_df = meta["Basel_Zuri_StainingPanel.csv"]
        # NOTE: uns_df = pd.read_csv(uns_path, skiprows=range(1, 10, 1))
        #   translates to skipping the first 9 rows in the read in csv file
        uns_df = uns_df.iloc[9:]
        # NOTE: now the following is no longer needed
        # uns_df = uns_df.dropna(axis=0, how="all")
        so.uns["staining_panel"] = uns_df

        # process OBS
        obs = self.gather_ct_data(labels, locs)
        obs = obs.set_index("CellId")
        obs.index.names = ["cell_id"]

        # select samples for which we have cell annotation in obs
        spl_df = spl_df.loc[obs.core.unique()]
        spls = spl_df.index.values

        # attr: spl
        so.spl = spl_df

        # Make map for numerica cell_type class
        keys = ["Stroma", "Immune", "Vessel", "Tumor"]
        numeric_labels = list(range(0, len(keys)))
        map_to_numeric = dict(zip(keys, numeric_labels))
        obs = obs.assign(class_id=obs["Class"].map(map_to_numeric))

        # process X
        X = self.split_scc_data(meta)
        X = X.set_index("core")
        colnames_X_wide = X.channel.unique()

        # Get metal tag labels
        unique_metal_tags = np.unique(uns_df["Metal Tag"].values)

        # Init dictionary
        map_to_metaltag = {}

        # Map matal tag label to column name
        for string in colnames_X_wide:
            for substring in unique_metal_tags:
                if substring in string:
                    map_to_metaltag[string] = substring

        for i, core in enumerate(spls):
            print(f"[{i}/{len(spls)}]\t{core}")
            grp_X = X.loc[core]

            # Load obs df
            grp_obs = obs[obs.core == core]
            grp_obs = grp_obs[
                [
                    "id",
                    "cluster",
                    "Cell_type",
                    "Class",
                    "class_id",
                    "PhenoGraph",
                    "Location_Center_X",
                    "Location_Center_Y",
                    "ObjectNumber",  # keep original object number, should be equivalent to id in mask?
                ]
            ]

            grp_X = grp_X.pivot(index="CellId", columns="channel", values="mc_counts")
            grp_X.index.names = ["cell_id"]

            assert len(grp_X) == len(grp_obs)
            assert len(set(grp_X.index).intersection(set(grp_obs.index))) == len(grp_X)

            # Join morphological features into obs
            grp_obs = grp_obs.join(
                grp_X[
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

            # tidy
            grp_obs.rename(str.lower, axis="columns", inplace=True)
            grp_obs.rename(columns={"class": "cell_class"}, inplace=True)
            grp_X = grp_X.rename(columns=map_to_metaltag)

            # Get metal tags present in sample
            metaltas_in_sampl = grp_X.columns.intersection(unique_metal_tags)

            # Drop other columns
            grp_X = grp_X[metaltas_in_sampl]

            so.X[core] = grp_X
            so.obs[core] = grp_obs

            # Load mask
            fname = Path(spl_df.loc[core]["FileName_FullStack"]).stem
            mask_fname = f"{fname}_maks.tiff"
            so.masks[core] = dict(cellmasks=masks[mask_fname])

        # # Drop metal tags only present in either of the cohorts
        metal_tags_in_all_spls = set(unique_metal_tags)
        for core in spls:
            metal_tags_in_all_spls = metal_tags_in_all_spls.intersection(
                so.X[core].columns
            )

        for core in spls:
            # Drop columns
            so.X[core] = so.X[core][list(metal_tags_in_all_spls)]

        ### Write to file ###
        with open(self.output, "wb") as f:
            pickle.dump(so, f)
