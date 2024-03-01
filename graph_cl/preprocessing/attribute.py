from pathlib import Path
import pandas as pd


def collect_features(processed_dir: Path, feat_config: dict):
    features = []
    for feat_name, feat_dict in feat_config.items():
        include = feat_dict.get("include", False)
        if include:
            feat_path = processed_dir / "features" / "observations" / feat_name
            feat = pd.read_parquet(feat_path)
            if isinstance(include, bool):
                features.append(feat)
            elif isinstance(include, list):
                features.append(feat[include])

    # add features
    feat = pd.concat(features, axis=1)
    assert feat.isna().any().any() == False

    return feat
