import os
import joblib
import json
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from typing import Union, List, Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np


def preprocess_and_save(df, cluster_col, categorical_cols, save_dir, n_quantiles_max=1000):
    os.makedirs(save_dir, exist_ok=True)

    # Save original features before any transformations
    df.to_csv(os.path.join(save_dir, "data_joint.csv"), index=False)

    # Save cluster column separately
    clusters = df[[cluster_col]].copy()
    clusters.to_csv(os.path.join(save_dir, "clusters.csv"), index=False)
    df = df.drop(columns=[cluster_col])

    # Identify numerical columns
    numerical_cols = df.columns.difference(categorical_cols).tolist()

    # Store number of classes per categorical variable BEFORE encoding
    num_classes_per_cat = [df[col].nunique() for col in categorical_cols]

    # Process categorical columns: one-hot encode (no log)
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False)
        X_cat = encoder.fit_transform(df[categorical_cols])
        joblib.dump(encoder, os.path.join(save_dir, "encoder.joblib"))

        cat_col_names = encoder.get_feature_names_out(categorical_cols)
        df_cat = pd.DataFrame(X_cat, columns=cat_col_names, index=df.index)
    else:
        df_cat = None
        print("No categorical columns found. Skipping one-hot encoding.")

    # Process numerical columns: quantile transform
    if numerical_cols:
        n_quantiles = min(n_quantiles_max, len(df))
        quantile_transformer = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution='normal', random_state=0
        )
        X_num = quantile_transformer.fit_transform(df[numerical_cols])
        joblib.dump(quantile_transformer, os.path.join(save_dir, "quantile_transformer.joblib"))

        df_num = pd.DataFrame(X_num, columns=numerical_cols, index=df.index)
    else:
        df_num = None
        print("No numerical columns found. Skipping quantile transformation.")

    # Combine processed numerical + categorical
    if df_num is not None and df_cat is not None:
        df_out = pd.concat([df_num, df_cat], axis=1)
    elif df_num is not None:
        df_out = df_num
    elif df_cat is not None:
        df_out = df_cat
    else:
        raise ValueError("No features to process!")

    # Save processed data
    df_out.to_csv(os.path.join(save_dir, "data_processed.csv"), index=False)

    # Save metadata
    num_features = len(numerical_cols) + len(categorical_cols)
    num_og_features = df_out.shape[1]
    num_samples = df_out.shape[0]
    f_s_ratio = round((num_features / num_samples) * 100, 3)

    metadata = {
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "cluster_col": cluster_col,
        "num_numerical_features": len(numerical_cols),
        "num_categorical_features": len(categorical_cols),
        "num_og_columns": num_features,
        "num_columns": num_og_features,
        "num_clusters": clusters[cluster_col].nunique(),
        "num_samples": num_samples,
        "f_s_ratio": f_s_ratio,
        "num_classes_per_cat": num_classes_per_cat
    }

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Data and metadata saved to '{save_dir}'")


def calculate_ranks(
    data: Union[str, List[Dict]],
    extra_data: Union[str, List[Dict]],
    metric_name: str,
    new_col_name: str,
    output_path: str
) -> pd.DataFrame:
    """
    Reads clustering results, appends an extra metric per dataset from another JSON,
    computes Avg. rank, Std. rank, Overall rank for method columns,
    and saves as JSON.

    Parameters
    ----------
    data : str | list[dict]
        - Main results JSON (path, string, or list of dicts).
    extra_data : str | list[dict]
        - Extra dataset info JSON (path, string, or list of dicts), must contain
          "dataset" matching the Data set ID and the metric to append.
    extra_metric_name : str
        - Name of the column to create for the extra metric.
    output_path : str
        - Path to save augmented JSON.

    Returns
    -------
    pd.DataFrame
        The augmented DataFrame.
    """

    # Helper to load JSON from various input types
    def _load_json(src):
        if isinstance(src, list):
            return src
        elif isinstance(src, str):
            if src.strip().startswith("["):  # JSON string
                return json.loads(src)
            else:  # file path
                with open(src, "r") as f:
                    return json.load(f)
        else:
            raise ValueError("Unsupported data type for JSON input.")

    # Load main data and extra data
    records = _load_json(data)
    extra_records = _load_json(extra_data)

    df = pd.DataFrame(records)
    extra_df = pd.DataFrame(extra_records)

    # Match "dataset" column in extra_df to "Data set ID" in df
    # Ensure both are same type for merge
    df["Data set ID"] = df["Data set ID"].astype(str)
    extra_df["dataset_index"] = extra_df["dataset_index"].astype(str)
    if metric_name == 'accuracy':
        extra_df[metric_name] = 100 * extra_df[metric_name]

    # Select the relevant column from extra_df
    if metric_name not in extra_df.columns:
        raise ValueError(f"'{metric_name}' not found in extra_data columns.")

    df = df.merge(
        extra_df[["dataset_index", metric_name]],
        left_on="Data set ID",
        right_on="dataset_index",
        how="left"
    ).drop(columns=["dataset_index"]).fillna(0)
    
    df = df.rename(columns={metric_name: new_col_name})

    # Identify method columns (exclude ID + extra metric column)
    id_col = "Data set ID"
    method_cols = [c for c in df.columns if c != id_col]

    # Rank within each dataset
    ranks = df[method_cols].round(1).rank(axis=1, ascending=False, method="average")

    # Summary stats
    avg_rank = ranks.mean(axis=0)
    std_rank = ranks.std(axis=0, ddof=1)
    overall_rank = avg_rank.rank(ascending=True, method="min")

    # Build summary rows
    summary_rows = pd.DataFrame(
        [
            ["Avg. rank", *avg_rank.tolist()],
            ["Std. rank", *std_rank.tolist()],
            ["Overall rank", *overall_rank.tolist()]
        ],
        columns=df.columns
    )

    # Append to table
    augmented_df = pd.concat([df, summary_rows], ignore_index=True)

    # Save
    augmented_df.to_json(output_path, orient="records", indent=2)

    return augmented_df


def keep_highest_accuracy(
    data: Union[str, List[Dict]],
    output_path: str
) -> pd.DataFrame:
    """
    For each dataset, keep only the entry with the highest accuracy.

    Parameters
    ----------
    data : str | list[dict]
        Path to JSON file, JSON string, or parsed list of dicts.
    output_path : str
        Path to save filtered JSON.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only the highest accuracy per dataset.
    """
    # Helper loader
    def _load_json(src):
        if isinstance(src, list):
            return src
        elif isinstance(src, str):
            if src.strip().startswith("["):  # JSON string
                return json.loads(src)
            else:  # file path
                with open(src, "r") as f:
                    return json.load(f)
        else:
            raise ValueError("Unsupported data type for JSON input.")

    # Load data
    records = _load_json(data)
    df = pd.DataFrame(records)

    # Ensure accuracy is numeric
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce").round(3)

    # Sort so that the highest accuracy is first per dataset
    df_sorted = df.sort_values(["dataset_index", "accuracy"], ascending=[True, False])

    # Keep first row for each dataset
    best_df = df_sorted.groupby("dataset_index", as_index=False).first()

    # Save to JSON
    best_df.to_json(output_path, orient="records", indent=2)

    return best_df


def calculate_ranks_multi(
    data: Union[str, List[Dict]],
    extras: List[Tuple[Union[str, List[Dict]], str, str]],
    output_path: str,
    id_col: str = "Data set ID",
    method_cols_override: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Reads main results, appends multiple extra metrics (from multiple JSONs),
    computes Avg. rank, Std. rank, Overall rank for method columns, and saves JSON.

    Parameters
    ----------
    data : str | list[dict]
        Main results JSON (path, JSON string, or list of dicts).
    extras : list[ (extra_data, metric_name, new_col_name) ]
        - extra_data: path | JSON string | list[dict] that must contain 'dataset_index' and metric
        - metric_name: column name in extra_data to pull
        - new_col_name: name to give that metric in the merged df
    output_path : str
        Path to save augmented JSON.
    id_col : str
        Column in main df used to match datasets (default "Data set ID").
    method_cols_override : list[str] | None
        If provided, use exactly these columns as method columns to rank (overrides detection).

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with summary rows appended.
    """

    # Helper to load JSON from various input types
    def _load_json(src):
        if isinstance(src, list):
            return src
        elif isinstance(src, str):
            if src.strip().startswith("["):  # JSON string
                return json.loads(src)
            else:  # file path
                with open(src, "r") as f:
                    return json.load(f)
        else:
            raise ValueError("Unsupported data type for JSON input.")

    # Load main data
    records = _load_json(data)
    df = pd.DataFrame(records)

    # Normalize id column to string for safe merges
    if id_col not in df.columns:
        raise ValueError(f"'{id_col}' not found in main data columns.")
    df[id_col] = df[id_col].astype(str)

    # Track names of new metric columns we add (for optional exclusion from ranking)
    added_metric_cols: List[str] = []

    # Merge each extra source
    for extra_data, metric_name, new_col_name in extras:
        extra_records = _load_json(extra_data)
        extra_df = pd.DataFrame(extra_records)

        # required join key in extras
        if "dataset_index" not in extra_df.columns:
            raise ValueError("'dataset_index' not found in extra_data columns.")
        extra_df["dataset_index"] = extra_df["dataset_index"].astype(str)

        # metric must be present
        if metric_name not in extra_df.columns:
            raise ValueError(f"'{metric_name}' not found in extra_data columns.")

        # optional: scale accuracy to %
        col_to_merge = extra_df[["dataset_index", metric_name]].copy()
        if metric_name == "accuracy":
            col_to_merge[metric_name] = 100 * col_to_merge[metric_name]

        # perform merge
        df = df.merge(
            col_to_merge,
            left_on=id_col,
            right_on="dataset_index",
            how="left"
        ).drop(columns=["dataset_index"])

        # rename and fill only the newly added column (avoid blanket fillna)
        df = df.rename(columns={metric_name: new_col_name})
        df[new_col_name] = df[new_col_name].fillna(0)
        added_metric_cols.append(new_col_name)

    # Decide which columns to rank
    if method_cols_override is not None:
        method_cols = method_cols_override
        missing = [c for c in method_cols if c not in df.columns]
        if missing:
            raise ValueError(f"method_cols_override columns not found: {missing}")
    else:
        # Start with all columns except the ID
        method_cols = [c for c in df.columns if c != id_col]
        # Optionally exclude the newly added extra metric columns

    if not method_cols:
        raise ValueError("No method columns selected for ranking.")

    # Rank within each dataset (row-wise), higher is better
    ranks = df[method_cols].round(1).rank(axis=1, ascending=False, method="average")

    # Summary stats (per column)
    avg_rank = ranks.mean(axis=0)
    std_rank = ranks.std(axis=0, ddof=1)
    overall_rank = avg_rank.rank(ascending=True, method="min")

    # Build summary rows aligned to full df columns
    def _summary_row(name, values_series):
        row = {id_col: name}
        row.update(values_series.to_dict())
        # Ensure all other columns exist; fill missing with None
        for c in df.columns:
            if c not in row:
                row[c] = None
        return row

    summary_rows = pd.DataFrame([
        _summary_row("Avg. rank", avg_rank),
        _summary_row("Std. rank", std_rank),
        _summary_row("Overall rank", overall_rank),
    ])[df.columns]  # keep column order

    augmented_df = pd.concat([df, summary_rows], ignore_index=True)

    # Save
    augmented_df.to_json(output_path, orient="records", indent=2)

    return augmented_df


def cluster_accuracy(y_true, y_pred):
    contingency = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = dict(zip(col_ind, row_ind))
    y_aligned = np.array([mapping[label] for label in y_pred])
    acc = accuracy_score(y_true, y_aligned)
    return acc, y_aligned
