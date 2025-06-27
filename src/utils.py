import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder

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
    num_features = df_out.shape[1]
    num_samples = df_out.shape[0]
    f_s_ratio = round((num_features / num_samples) * 100, 3)

    metadata = {
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "cluster_col": cluster_col,
        "num_numerical_features": len(numerical_cols),
        "num_categorical_features": len(categorical_cols),
        "num_columns": num_features,
        "num_clusters": clusters[cluster_col].nunique(),
        "num_samples": num_samples,
        "f_s_ratio": f_s_ratio,
        "num_classes_per_cat": num_classes_per_cat
    }

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Data and metadata saved to '{save_dir}'")
