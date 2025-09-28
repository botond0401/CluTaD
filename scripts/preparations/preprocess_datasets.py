import os
import json
import pandas as pd
from scipy.io import arff
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils import preprocess_and_save, balance_two_cluster_dataset


def main():
    config_path = "data/misc/dataset_config.json"
    raw_dir = "data/raw"
    output_dir = "data/preprocessed"
    balanced_output_dir = "data/preprocessed_balanced"

    with open(config_path, "r") as f:
        configs = json.load(f)

    for dataset_id, cfg in configs.items():
        file_path = os.path.join(raw_dir, cfg["file"])
        cluster_col = cfg["cluster_col"]
        categorical_cols = cfg["categorical_cols"]

        print(f"🔄 Loading dataset {dataset_id}...")
        data, _ = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        df = df.map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

        save_dir = os.path.join(output_dir, dataset_id)
        preprocess_and_save(df, cluster_col, categorical_cols, save_dir)

        df_balanced = balance_two_cluster_dataset(df, cluster_col)
        if df_balanced is not None:
            balanced_save_dir = os.path.join(balanced_output_dir, f"{dataset_id}b")
            print(f"⚖️  Creating balanced version of dataset {dataset_id}...")
            preprocess_and_save(df_balanced, cluster_col, categorical_cols, balanced_save_dir)


if __name__ == "__main__":
    main()
