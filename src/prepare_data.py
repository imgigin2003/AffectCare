import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(processed_path="../data/processed", test_size=0.2):
    X = np.load(f"{processed_path}/X.npy")
    y = np.load(f"{processed_path}/y.npy")
    # X shape: (N, 1, 13, 173)
    # y shape: (N,)

    # Step 1 -> Flattening the Array
    # StandardScaler only understands flat 2D arrays — (N, features).
    # So we flatten each (1, 13, 173) into one long row of numbers:
    # -1 tells numpy: "calculate the rest automatically"
    # (N, 1×13×173) = (N, 2249)
    N = X.shape[0]
    X_flat = X.reshape(N, -1)

    # Step 2 -> Splitting training set from test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat,
        y,
        test_size=test_size,  # 20% for testing, 80% for training
        random_state=42,  # makes the split reproducible every run
        stratify=y,  # keeps distress/normal ratio equal in both splits
    )
    stratify = y  # is your Recall-First safeguard here — without it,
    # by bad luck the test set could end up with zero distress samples.

    # Step 3 -> Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train
    )  # fit_transform on training, transform only on test.
    X_test = scaler.transform(X_test)
    # Refitting on test data is the "moving goalposts" bug

    return X_train, X_test, y_train, y_test, scaler
