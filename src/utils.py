import os
import joblib
import json
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder


def preprocess_and_save(df, cluster_col, categorical_cols, save_dir, n_quantiles_max = 1000):
    os.makedirs(save_dir, exist_ok=True)

    # Drop cluster column and save separately
    clusters = df[[cluster_col]].copy()
    clusters.to_csv(os.path.join(save_dir, "clusters.csv"), index=False)
    df = df.drop(columns=[cluster_col])

    # Identify numerical columns
    numerical_cols = df.columns.difference(categorical_cols).tolist()

    # Store number of classes per categorical variable BEFORE encoding
    num_classes_per_cat = [df[col].nunique() for col in categorical_cols]

    if categorical_cols:
        encoder = OrdinalEncoder()
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
        joblib.dump(encoder, os.path.join(save_dir, "encoder.joblib"))
    else:
        print("No categorical columns found. Skipping ordinal encoding.")

    if numerical_cols:
        n_quantiles = min(n_quantiles_max, len(df))

        quantile_transformer = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution='normal', random_state=0
        )
        df[numerical_cols] = quantile_transformer.fit_transform(df[numerical_cols])
        joblib.dump(quantile_transformer, os.path.join(save_dir, "quantile_transformer.joblib"))
    else:
        print("No numerical columns found. Skipping quantile transformation.")

    # Reorder columns: numerical first, categorical last
    df = df[numerical_cols + categorical_cols]

    # Save processed data
    df.to_csv(os.path.join(save_dir, "data_processed.csv"), index=False)

    # Compute metadata
    num_features = len(df.columns)
    num_samples = len(df)
    f_s_ratio = round((num_features / num_samples) * 100, 3)

    # Save metadata
    metadata = {
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "cluster_col": cluster_col,
        "num_numerical_features": len(numerical_cols),
        "num_categorical_features": len(categorical_cols),
        "num_features": num_features,
        "num_clusters": clusters[cluster_col].nunique(),
        "num_samples": num_samples,
        "f_s_ratio": f_s_ratio,
        "num_classes_per_cat": num_classes_per_cat
    }

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Data and metadata saved to '{save_dir}'")
