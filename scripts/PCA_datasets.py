import os
import json
import pandas as pd
from scipy.io import arff
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import preprocess_and_save


def apply_pca(df, n_components=4, exclude_cols=None):
    """Apply PCA to dataframe, keeping cluster column untouched."""
    if exclude_cols is None:
        exclude_cols = []

    # Separate features and excluded columns
    features = df.drop(columns=exclude_cols)
    to_keep = df[exclude_cols]

    # Standardize
    X_scaled = StandardScaler().fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Build new DataFrame
    df_pca = pd.DataFrame(X_pca, columns=[f"PCA{i+1}" for i in range(n_components)])
    df_out = pd.concat([df_pca, to_keep.reset_index(drop=True)], axis=1)

    print(f"✅ Applied PCA: explained variance ratios = {pca.explained_variance_ratio_}")
    return df_out


def main():
    config_path = "data/misc/dataset_config.json"
    raw_dir = "data/raw"
    pca_output_dir = "data/preprocessed_pca"

    with open(config_path, "r") as f:
        configs = json.load(f)

    # Only process dataset 458
    dataset_id = "458"
    cfg = configs[dataset_id]

    file_path = os.path.join(raw_dir, cfg["file"])
    cluster_col = cfg["cluster_col"]

    print(f"🔄 Loading dataset {dataset_id}...")
    data, _ = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    df = df.map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    # Apply PCA and save
    pca_save_dir = os.path.join(pca_output_dir, f"{dataset_id}_pca")
    print(f"📉 Applying PCA (4 components) to dataset {dataset_id}...")
    df_pca = apply_pca(df, n_components=4, exclude_cols=[cluster_col])
    preprocess_and_save(df_pca, cluster_col, [], pca_save_dir)


if __name__ == "__main__":
    main()
