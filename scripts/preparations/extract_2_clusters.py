import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def main():
    config_path = "data/misc/dataset_config.json"
    input_dir = "data/preprocessed"
    output_dir = "data/misc"

    with open(config_path, "r") as f:
        configs = json.load(f)

    datasets_w_2_clusters = []
    for dataset_id, _ in configs.items():
        print(f"🔄 Processing dataset {dataset_id}...")

        METADATA_PATH = os.path.join(input_dir, dataset_id, 'metadata.json')
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        n_clusters = metadata['num_clusters']
        
        if n_clusters == 2:
            datasets_w_2_clusters.append(dataset_id)

    OUTPUT_PATH = os.path.join(output_dir, "datasets_w_2_clusters.json")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(datasets_w_2_clusters, f, indent=4)

    print(f"✅ Data and metadata saved to '{OUTPUT_PATH}'")

if __name__ == "__main__":
    main()