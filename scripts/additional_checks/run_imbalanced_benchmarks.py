import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils import cluster_accuracy


def run_best_of_n(model_class, X, y_true, n_runs=10, **kwargs):
    best_acc, best_ari = -1, -1
    for seed in range(n_runs):
        model = model_class(n_clusters=2, random_state=seed, n_init=10, **kwargs) \
            if model_class == KMeans else model_class(n_components=2, random_state=seed, n_init=10, **kwargs)
        y_pred = model.fit_predict(X)
        acc, _ = cluster_accuracy(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        if acc > best_acc:
            best_acc, best_ari = acc, ari
    return best_acc, best_ari

def main():
    config_path = "data/misc/dataset_config.json"
    input_dir = "data/preprocessed"
    output_dir = "data/results"

    with open(config_path, "r") as f:
        configs = json.load(f)

    # results tables
    acc_results = pd.DataFrame(columns=["dataset", "kmeans", "gmm"])
    ari_results = pd.DataFrame(columns=["dataset", "kmeans", "gmm"])

    for dataset_id, _ in configs.items():
        print(f"🔄 Processing dataset {dataset_id}...")

        dataset_id = f"{dataset_id}b"
        dataset_path = os.path.join(input_dir, dataset_id)
        data_file = os.path.join(dataset_path, "data_processed.csv")
        labels_file = os.path.join(dataset_path, "clusters.csv")

        if not (os.path.exists(data_file) and os.path.exists(labels_file)):
            print(f"⚠️ Skipping {dataset_id}, missing files.")
            continue

        X = pd.read_csv(data_file)
        y_true = pd.read_csv(labels_file).values.flatten()
        if y_true.dtype.kind in {'U', 'S', 'O'}:
            _, y_true = np.unique(np.asarray(y_true).astype(str), return_inverse=True)

        acc_kmeans, ari_kmeans = run_best_of_n(KMeans, X, y_true, n_runs=10)

        acc_gmm, ari_gmm = run_best_of_n(GaussianMixture, X, y_true, n_runs=10)

        # save results
        acc_results.loc[len(acc_results)] = [dataset_id, acc_kmeans, acc_gmm]
        ari_results.loc[len(ari_results)] = [dataset_id, ari_kmeans, ari_gmm]

    # save result tables
    acc_results.to_csv(os.path.join(output_dir, "clutad/balanced/balanced_clustering_accuracies.csv"), index=False)
    ari_results.to_csv(os.path.join(output_dir, "clutad/balanced/balanced_clustering_ari.csv"), index=False)

    print("✅ Finished. Results saved to:")
    print(f"   - {os.path.join(output_dir, 'clutad/balanced/balanced_clustering_accuracies.csv')}")
    print(f"   - {os.path.join(output_dir, 'clutad/balanced/balanced_clustering_ari.csv')}")

if __name__ == "__main__":
    main()
