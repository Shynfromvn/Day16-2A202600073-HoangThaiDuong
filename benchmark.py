import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATASET_PATH = Path("creditcard.csv")
RESULT_PATH = Path("benchmark_result.json")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "creditcard.csv not found. Download and unzip the Kaggle dataset first."
        )

    load_start = time.perf_counter()
    df = pd.read_csv(DATASET_PATH)
    load_time = time.perf_counter() - load_start

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - train_start

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    one_row = X_test.iloc[[0]]
    single_runs = 200
    single_start = time.perf_counter()
    for _ in range(single_runs):
        model.predict_proba(one_row)
    single_total = time.perf_counter() - single_start
    single_latency_ms = (single_total / single_runs) * 1000

    batch_size = min(1000, len(X_test))
    batch = X_test.iloc[:batch_size]
    batch_start = time.perf_counter()
    model.predict_proba(batch)
    batch_total = time.perf_counter() - batch_start
    throughput_rows_per_sec = batch_size / batch_total if batch_total > 0 else float("inf")

    booster = model.booster_
    best_iteration = booster.current_iteration() if booster is not None else model.n_estimators

    results = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "load_time_seconds": round(load_time, 4),
        "training_time_seconds": round(training_time, 4),
        "best_iteration": int(best_iteration),
        "auc_roc": round(float(auc), 6),
        "accuracy": round(float(accuracy), 6),
        "f1_score": round(float(f1), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "inference_latency_single_row_ms": round(float(single_latency_ms), 6),
        "inference_throughput_1000_rows_per_sec": round(float(throughput_rows_per_sec), 6),
    }

    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("LightGBM CPU Benchmark Complete")
    print(f"Dataset rows: {results['dataset_rows']}")
    print(f"Load time: {results['load_time_seconds']} seconds")
    print(f"Training time: {results['training_time_seconds']} seconds")
    print(f"Best iteration: {results['best_iteration']}")
    print(f"AUC-ROC: {results['auc_roc']}")
    print(f"Accuracy: {results['accuracy']}")
    print(f"F1-Score: {results['f1_score']}")
    print(f"Precision: {results['precision']}")
    print(f"Recall: {results['recall']}")
    print(
        "Inference latency (1 row): "
        f"{results['inference_latency_single_row_ms']} ms"
    )
    print(
        "Inference throughput (1000 rows): "
        f"{results['inference_throughput_1000_rows_per_sec']} rows/sec"
    )
    print(f"Saved results to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
