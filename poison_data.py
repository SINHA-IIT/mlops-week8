import pandas as pd
import numpy as np
import sys
import os

"""
Usage:
    python poison_data.py <poison_fraction>

Example:
    python poison_data.py 0.10
"""

def poison_features(df, frac):
    df = df.copy()
    n = len(df)
    k = int(frac * n)

    if k == 0:
        return df

    # randomly select rows to poison
    idx = np.random.choice(n, size=k, replace=False)

    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    for col in feature_cols:
        mu = df[col].mean()
        sigma = df[col].std()
        if sigma == 0 or np.isnan(sigma):
            sigma = 1.0
        # Add strong noise ×3 std deviation
        df.loc[idx, col] = np.random.normal(mu, sigma * 3, size=k)

    return df


def main():
    if len(sys.argv) != 2:
        print("Usage: python poison_data.py <poison_fraction>")
        sys.exit(1)

    frac = float(sys.argv[1])
    print(f"🔥 Applying FEATURE POISONING with fraction {frac}")

    df = pd.read_csv("data/data_iris.csv")
    poisoned_df = poison_features(df, frac)
    poisoned_df.to_csv("data/data_iris_poisoned.csv", index=False)

    print(f"✅ Saved poisoned dataset to data/data_iris_poisoned.csv")


if __name__ == "__main__":
    main()
