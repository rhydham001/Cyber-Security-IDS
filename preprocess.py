"""
NSL-KDD Dataset Preprocessing Pipeline
Handles loading, encoding, scaling, and label mapping for the IDS model.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# ---------------------------------------------------------------------------
# NSL-KDD column definitions
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Map specific attack names → 5 categories
ATTACK_MAP = {
    "normal": "Normal",
    # DoS attacks
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
    # Probe attacks
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L attacks
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "snmpgetattack": "R2L", "named": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "sendmail": "R2L", "httptunnel": "R2L",
    "snmpguess": "R2L",
    # U2R attacks
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "xterm": "U2R", "ps": "U2R", "sqlattack": "U2R",
    "httptunnel": "R2L",
}

ATTACK_CATEGORIES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load a NSL-KDD .txt file (headerless CSV)."""
    df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
    df.drop(columns=["difficulty"], inplace=True)
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map specific attack names to 5-class categories."""
    df["label"] = df["label"].str.strip().str.lower()
    df["attack_category"] = df["label"].map(ATTACK_MAP).fillna("Unknown")
    # Drop any rows with unmapped labels
    df = df[df["attack_category"] != "Unknown"].copy()
    df.drop(columns=["label"], inplace=True)
    return df


def encode_and_scale(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fit: bool = True,
):
    """
    One-hot encode categorical features and MinMax-scale numeric features.
    Returns (X_train, y_train, X_test, y_test) as numpy arrays.
    Saves fitted encoders/scalers to MODELS_DIR when fit=True.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Separate features and labels ---
    y_train_raw = train_df["attack_category"]
    y_test_raw = test_df["attack_category"]
    X_train = train_df.drop(columns=["attack_category"])
    X_test = test_df.drop(columns=["attack_category"])

    # --- One-hot encode categoricals ---
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, dtype=int)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLS, dtype=int)

    # Align columns (train may have categories not in test and vice versa)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Save column order for inference
    feature_columns = X_train.columns.tolist()

    # --- Scale numeric features ---
    if fit:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    else:
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    # --- Encode labels ---
    if fit:
        le = LabelEncoder()
        le.fit(ATTACK_CATEGORIES)
        joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    else:
        le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    # Save feature column names
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.pkl"))

    return X_train_scaled, y_train, X_test_scaled, y_test, feature_columns


def preprocess_single(raw_features: dict) -> np.ndarray:
    """
    Preprocess a single sample dict (41 raw features) for prediction.
    Uses saved scaler and column order.
    """
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

    df = pd.DataFrame([raw_features])
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, dtype=int)

    # Align to training feature columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(df)
    return scaled


def get_preprocessed_data():
    """Full pipeline: load → map → encode → scale. Returns train/test arrays."""
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    train_df = map_labels(train_df)
    test_df = map_labels(test_df)

    return encode_and_scale(train_df, test_df, fit=True), train_df, test_df


if __name__ == "__main__":
    (X_train, y_train, X_test, y_test, feat_cols), _, _ = get_preprocessed_data()
    print(f"Training set: {X_train.shape}  |  Test set: {X_test.shape}")
    print(f"Classes: {ATTACK_CATEGORIES}")
    print(f"Feature count: {len(feat_cols)}")
