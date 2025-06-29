import os
import json
import pandas as pd
from scipy.io import arff
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import preprocess_and_save


def main():
    config_path = "data/misc/dataset_config.json"
    raw_dir = "data/raw"
    output_dir = "data/preprocessed"

    with open(config_path, "r") as f:
        configs = json.load(f)

    for dataset_id, cfg in configs.items():
        file_path = os.path.join(raw_dir, cfg["file"])
        cluster_col = cfg["cluster_col"]
        categorical_cols = cfg["categorical_cols"]
        save_dir = os.path.join(output_dir, dataset_id)

        print(f"🔄 Processing dataset {dataset_id}...")
        
        data, _ = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        df = df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        preprocess_and_save(df, cluster_col, categorical_cols, save_dir)

if __name__ == "__main__":
    main()
