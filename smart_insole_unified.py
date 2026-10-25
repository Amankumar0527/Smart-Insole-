"""
Smart Insole: Unified ANN Training + Prediction + Optional Firebase Listener
=============================================================================

What this version does:
1. Trains/evaluates the standing-vs-walking ANN from the labelled Excel dataset.
2. Saves the complete scikit-learn Pipeline (scaler + ANN) in one model file.
3. Predicts standing/walking from live four-FSR readings.
4. Optionally listens to Firebase Realtime Database and writes activity results back.

IMPORTANT SCOPE:
- The current training dataset contains four FSR readings and a Condition label.
  Therefore, this trained model uses ONLY the four FSR values.
- IMU, temperature/humidity, height and weight are not model inputs in this
  version because the existing training dataset does not provide aligned,
  labelled examples for them.
- This code does NOT determine medically normal/abnormal gait, estimate GRF/CoP,
  or generate corrective walking advice. Those require separate reference-labelled
  datasets and validation.
- Recording groups inferred from timestamps are only approximate. Prefer
  Participant ID and Recording ID and hold out complete participants for testing.

Install:
    python -m pip install pandas openpyxl scikit-learn joblib firebase-admin

Train and evaluate:
    python smart_insole.py train --data "Smart_Insole_Readings_Grouped.xlsx"

Predict one sample:
    python smart_insole.py predict --fsr1 120 --fsr2 200 --fsr3 160 --fsr4 90

Optional Firebase Realtime Database listener:
    1. Set environment variable FIREBASE_DATABASE_URL to your RTDB URL.
    2. Set GOOGLE_APPLICATION_CREDENTIALS to your Firebase service-account JSON path.
    3. Put live readings at /live_samples/latest as an object containing:
       {"FSR 1": 120, "FSR 2": 200, "FSR 3": 160, "FSR 4": 90,
        "temperature_c": 30.2, "humidity_pct": 55, "acc_x": 0.1, ...}
    4. Run:
       python smart_insole.py firebase --model ann_results/smart_insole_ann.joblib

    Results are written to /analysis/latest. Adapt database paths and field names
    to match your ESP32 firmware and Firebase structure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["FSR 1", "FSR 2", "FSR 3", "FSR 4"]
TARGET = "Condition"
VALID_LABELS = {"standing", "walking"}
DEFAULT_MODEL = Path("ann_results") / "smart_insole_ann.joblib"


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common dataset column names."""
    rename = {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]+", "", str(col).lower())
        if key in {"fsr1", "sensor1", "fsrtoe"}:
            rename[col] = "FSR 1"
        elif key in {"fsr2", "sensor2", "fsrmet1"}:
            rename[col] = "FSR 2"
        elif key in {"fsr3", "sensor3", "fsrmet5"}:
            rename[col] = "FSR 3"
        elif key in {"fsr4", "sensor4", "fsrheel"}:
            rename[col] = "FSR 4"
        elif key in {"condition", "activity", "label", "activitylabel"}:
            rename[col] = "Condition"
        elif key in {"serialtime", "serialmonitortime"}:
            rename[col] = "Serial Time"
        elif key in {"timestampms", "timestamp", "timestampmillisecond"}:
            rename[col] = "Timestamp (ms)"
        elif key in {"weightkg", "weight"}:
            rename[col] = "Weight (kg)"
        elif key in {"participantid", "participant", "personid", "subjectid"}:
            rename[col] = "Participant ID"
        elif key in {"recordingid", "sessionid", "trialid"}:
            rename[col] = "Recording ID"
    return df.rename(columns=rename)


def parse_serial_time(value: Any) -> float:
    """Convert HH:MM:SS.mmm to seconds since midnight."""
    if pd.isna(value):
        return float("nan")
    match = re.match(r"^\s*(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*$", str(value))
    if not match:
        return float("nan")
    hh, mm, ss, frac = match.groups()
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + (float("0." + frac) if frac else 0.0)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Put the Excel workbook beside this script or pass --data with its path."
        )
    xls = pd.ExcelFile(path)
    sheet = "All Readings" if "All Readings" in xls.sheet_names else xls.sheet_names[0]
    df = normalize_column_names(pd.read_excel(path, sheet_name=sheet))

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in '{sheet}': {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    df = df[df[TARGET].isin(VALID_LABELS)].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    if df.empty or set(df[TARGET].unique()) != VALID_LABELS:
        raise ValueError(
            "Need valid FSR 1–4 readings and both labels: 'standing' and 'walking'."
        )

    if "Serial Time" in df.columns:
        df["_time_seconds"] = df["Serial Time"].apply(parse_serial_time)
    else:
        df["_time_seconds"] = np.nan
    return df


def make_recording_groups(df: pd.DataFrame) -> pd.Series:
    """Use explicit participant/recording IDs when present; otherwise infer groups."""
    if "Participant ID" in df.columns and "Recording ID" in df.columns:
        return (
            df["Participant ID"].astype(str) + "__" + df["Recording ID"].astype(str)
        )

    groups = pd.Series(index=df.index, dtype="object")
    category_cols = [TARGET]
    if "Weight (kg)" in df.columns:
        category_cols.append("Weight (kg)")
    group_num = 0
    has_time = df["_time_seconds"].notna().any()

    for _, indices_obj in df.groupby(category_cols, sort=False).groups.items():
        indices = list(indices_obj)
        valid_times = df.loc[indices, "_time_seconds"].notna().all()
        if has_time and valid_times:
            times = df.loc[indices, "_time_seconds"].to_numpy()
            starts = [0]
            for i in range(1, len(times)):
                delta = times[i] - times[i - 1]
                if delta < 0 or delta > 1.5:
                    starts.append(i)
            starts.append(len(indices))
            for start, end in zip(starts[:-1], starts[1:]):
                group_num += 1
                groups.loc[indices[start:end]] = f"inferred_recording_{group_num}"
        else:
            for start in range(0, len(indices), 25):
                group_num += 1
                groups.loc[indices[start:start + 25]] = f"inferred_block_{group_num}"
    return groups.fillna("inferred_unknown")


def split_data(X, y, groups, test_size: float, random_state: int):
    """Prefer group split; warn and use row split only if no viable group split exists."""
    splitter = GroupShuffleSplit(
        n_splits=100, test_size=test_size, random_state=random_state
    )
    for train_idx, test_idx in splitter.split(X, y, groups):
        if set(y.iloc[train_idx].unique()) == VALID_LABELS and set(y.iloc[test_idx].unique()) == VALID_LABELS:
            return train_idx, test_idx, "group split"

    print(
        "WARNING: Could not find a group split containing both classes. "
        "Using stratified row split; accuracy may be optimistic.",
        file=sys.stderr,
    )
    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return train_idx, test_idx, "stratified row split (fallback)"


def train(args):
    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    groups = make_recording_groups(df)
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(str)

    train_idx, test_idx, split_description = split_data(
        X, y, groups, args.test_size, args.random_state
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Pipeline ensures StandardScaler is fitted on training data only.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ann", MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=1500,
            early_stopping=False,
            n_iter_no_change=30,
            random_state=args.random_state,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    labels = ["standing", "walking"]
    report = classification_report(y_test, predictions, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=labels)

    # Save the full pipeline so live prediction uses the exact same scaler.
    model_path = output_dir / "smart_insole_ann.joblib"
    joblib.dump({
        "model": model,
        "features": FEATURES,
        "labels": labels,
        "purpose": "standing_vs_walking_from_four_FSR_readings",
    }, model_path)

    (output_dir / "classification_report.txt").write_text(
        "SMART INSOLE ANN — STANDING VS WALKING\n"
        "======================================\n"
        f"Dataset: {data_path.name}\n"
        f"Rows used: {len(df)}\n"
        f"Train rows: {len(train_idx)}\n"
        f"Test rows: {len(test_idx)}\n"
        f"Group count: {groups.nunique()}\n"
        f"Split method: {split_description}\n"
        f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
        f"Class counts:\n{y.value_counts().to_string()}\n\n"
        f"Classification report:\n{report}\n"
        f"Confusion matrix (actual rows, predicted columns):\n"
        f"{pd.DataFrame(cm, index=['Actual standing', 'Actual walking'], columns=['Predicted standing', 'Predicted walking']).to_string()}\n\n"
        "Note: inferred groups are not a substitute for participant-independent testing. "
        "This classifier does not assess medically normal/abnormal gait.\n",
        encoding="utf-8",
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title("Smart Insole ANN — Confusion Matrix")
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipping confusion-matrix image.")

    test_results = df.iloc[test_idx].copy()
    test_results["Predicted Condition"] = predictions
    test_results["Correct Prediction"] = test_results[TARGET].to_numpy() == predictions
    test_results.drop(columns=["_time_seconds"], errors="ignore").to_csv(
        output_dir / "test_predictions.csv", index=False
    )

    print("\nSMART INSOLE ANN — TRAINING COMPLETE")
    print(f"Valid rows: {len(df)} | approximate groups: {groups.nunique()}")
    print(f"Split: {split_description}")
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:\n", report)
    print(f"Saved model: {model_path}")
    print(f"Saved report and test predictions in: {output_dir}")
    print("Scope: standing/walking classification only; not a gait diagnosis.")


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run the train mode first."
        )
    bundle = joblib.load(model_path)
    return bundle["model"], bundle.get("features", FEATURES), bundle.get("labels", ["standing", "walking"])


def normalize_live_sample(payload: dict) -> pd.DataFrame:
    """Read a single live sensor object with common FSR field-name variants."""
    aliases = {
        "FSR 1": ["FSR 1", "fsr1", "fsr_toe", "FSR_1"],
        "FSR 2": ["FSR 2", "fsr2", "fsr_met1", "FSR_2"],
        "FSR 3": ["FSR 3", "fsr3", "fsr_met5", "FSR_3"],
        "FSR 4": ["FSR 4", "fsr4", "fsr_heel", "FSR_4"],
    }
    values = {}
    for field, options in aliases.items():
        found = next((payload[k] for k in options if k in payload), None)
        if found is None:
            raise ValueError(f"Missing live sensor field for {field}; accepted names: {options}")
        values[field] = float(found)
    return pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)


def predict_one(model, payload: dict) -> dict:
    X_live = normalize_live_sample(payload)
    predicted = str(model.predict(X_live)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_live)[0]
        confidence = float(np.max(probabilities))

    return {
        "activity": predicted,
        "model_confidence": confidence,
        "gait_status": "not_assessed",
        "recommendations": [],
        "note": (
            "This model classifies standing/walking from four FSR values only. "
            "Gait quality and personalised corrective advice are not validated."
        ),
        "processed_at_unix": time.time(),
    }


def predict_command(args):
    model, _, _ = load_model(Path(args.model))
    payload = {
        "FSR 1": args.fsr1,
        "FSR 2": args.fsr2,
        "FSR 3": args.fsr3,
        "FSR 4": args.fsr4,
    }
    print(json.dumps(predict_one(model, payload), indent=2))


def firebase_command(args):
    """Listen to a Firebase RTDB node and write the latest activity result."""
    try:
        import firebase_admin
        from firebase_admin import credentials, db
    except ImportError as exc:
        raise RuntimeError(
            "Firebase SDK missing. Install it with: pip install firebase-admin"
        ) from exc

    database_url = os.environ.get("FIREBASE_DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set FIREBASE_DATABASE_URL to your Firebase Realtime Database URL.")

    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise RuntimeError(
            "Set GOOGLE_APPLICATION_CREDENTIALS to your Firebase service-account JSON path."
        )

    model, _, _ = load_model(Path(args.model))
    cred = credentials.Certificate(credentials_path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {"databaseURL": database_url})

    input_ref = db.reference(args.input_path)
    output_ref = db.reference(args.output_path)

    print(f"Listening to Firebase path: {args.input_path}")
    print(f"Writing predictions to: {args.output_path}")
    print("Press Ctrl+C to stop.")

    def on_event(event):
        try:
            data = event.data
            # The configured input node should contain one object with four FSR values.
            if not isinstance(data, dict):
                return
            prediction = predict_one(model, data)
            output_ref.set(prediction)
            print(json.dumps(prediction, indent=2))
        except Exception as exc:
            error = {
                "status": "processing_error",
                "message": str(exc),
                "processed_at_unix": time.time(),
            }
            print(json.dumps(error), file=sys.stderr)

    stream = input_ref.listen(on_event)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stream.close()
        print("\nFirebase listener stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Smart Insole ANN training and live prediction"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train/evaluate ANN from Excel")
    train_parser.add_argument("--data", default="Smart_Insole_Readings_Grouped.xlsx")
    train_parser.add_argument("--output", default="ann_results")
    train_parser.add_argument("--test-size", type=float, default=0.25)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict", help="Predict one four-FSR sample")
    predict_parser.add_argument("--model", default=str(DEFAULT_MODEL))
    predict_parser.add_argument("--fsr1", type=float, required=True)
    predict_parser.add_argument("--fsr2", type=float, required=True)
    predict_parser.add_argument("--fsr3", type=float, required=True)
    predict_parser.add_argument("--fsr4", type=float, required=True)
    predict_parser.set_defaults(func=predict_command)

    firebase_parser = subparsers.add_parser("firebase", help="Listen to Firebase RTDB")
    firebase_parser.add_argument("--model", default=str(DEFAULT_MODEL))
    firebase_parser.add_argument("--input-path", default="/live_samples/latest")
    firebase_parser.add_argument("--output-path", default="/analysis/latest")
    firebase_parser.set_defaults(func=firebase_command)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
