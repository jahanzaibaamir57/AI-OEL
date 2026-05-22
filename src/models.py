import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

from preprocessing import DataPreprocessor, build_data_dictionary


SEED = 42


def _ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _save_feature_importance_plot(model: DecisionTreeClassifier, feature_names: np.ndarray, path: Path) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_names))

    plt.figure(figsize=(11, 6))
    plt.bar(range(top_n), importances[order][:top_n], color="#2a9d8f")
    plt.xticks(range(top_n), feature_names[order][:top_n], rotation=55, ha="right")
    plt.title("Decision Tree Feature Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_cluster_plot(X_scaled: np.ndarray, labels: np.ndarray, path: Path) -> None:
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=10, cmap="viridis", alpha=0.7)
    plt.title("KNN-Based Soil Zone Clusters (PCA View)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, s=12, alpha=0.7, color="#e76f51")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Linear Regression Residual Analysis")
    plt.xlabel("Predicted Yield")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _build_cluster_guidance(df: pd.DataFrame, labels: np.ndarray) -> Dict[str, str]:
    guidance = {}
    with_labels = df.copy()
    with_labels["cluster"] = labels

    for cluster_id, grp in with_labels.groupby("cluster"):
        rain = grp["average_rain_fall_mm_per_year"].mean()
        temp = grp["avg_temp"].mean()
        pest = grp["pesticides_tonnes"].mean()

        suggestion = "Balanced zone: maintain current irrigation and monitoring schedule."
        if rain < with_labels["average_rain_fall_mm_per_year"].median():
            suggestion = "Lower rainfall zone: prioritize moisture conservation and drip irrigation."
        if temp > with_labels["avg_temp"].quantile(0.7):
            suggestion = "Warmer zone: use heat-resilient crop management and mulching."
        if pest > with_labels["pesticides_tonnes"].quantile(0.7):
            suggestion = "High pesticide load zone: optimize integrated pest management practices."

        guidance[str(int(cluster_id))] = suggestion

    return guidance


def train_all(
    data_path: Path,
    models_dir: Path,
    results_dir: Path,
    random_state: int = SEED,
) -> Dict[str, float]:
    _ensure_dirs(models_dir, results_dir)

    pre = DataPreprocessor()
    raw_df = pre.load_data(data_path)
    clean_df = pre.fit_transform(raw_df)

    clean_df.to_csv(results_dir / "cleaned_dataset.csv", index=False)
    build_data_dictionary(clean_df).to_csv(results_dir / "data_dictionary.csv", index=False)
    pre.save_metadata(results_dir)

    # Classification: crop recommendation
    X_cls = clean_df[["Area", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]]
    y_cls = clean_df["Item"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=random_state, stratify=y_cls
    )

    cls_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Area"]),
            (
                "num",
                "passthrough",
                ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"],
            ),
        ]
    )

    dt_pipeline = Pipeline(
        steps=[
            ("prep", cls_preprocessor),
            (
                "model",
                DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=random_state),
            ),
        ]
    )
    dt_pipeline.fit(Xc_train, yc_train)
    yc_pred = dt_pipeline.predict(Xc_test)

    cls_accuracy = accuracy_score(yc_test, yc_pred)
    cls_precision = precision_score(yc_test, yc_pred, average="weighted", zero_division=0)
    cls_recall = recall_score(yc_test, yc_pred, average="weighted", zero_division=0)

    trained_dt = dt_pipeline.named_steps["model"]
    prep = dt_pipeline.named_steps["prep"]
    feature_names = prep.get_feature_names_out()
    _save_feature_importance_plot(
        trained_dt,
        feature_names,
        results_dir / "feature_importance.png",
    )

    # Clustering with KMeans + KNN inference layer
    cluster_features = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
    X_cluster = clean_df[cluster_features].copy()
    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

    best_k = 4
    best_score = -1.0
    best_labels = None
    best_model = None
    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_cluster_scaled)
        score = silhouette_score(X_cluster_scaled, labels)
        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels
            best_model = km

    knn_cluster_inference = KNeighborsClassifier(n_neighbors=7)
    knn_cluster_inference.fit(X_cluster_scaled, best_labels)

    _save_cluster_plot(X_cluster_scaled, best_labels, results_dir / "cluster_scatter.png")

    cluster_guidance = _build_cluster_guidance(clean_df, best_labels)

    # Regression: yield prediction
    X_reg = clean_df[["Area", "Item", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]]
    y_reg = clean_df["hg/ha_yield"]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=random_state
    )

    reg_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Area", "Item"]),
            (
                "num",
                StandardScaler(),
                ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"],
            ),
        ]
    )

    reg_pipeline = Pipeline(
        steps=[
            ("prep", reg_preprocessor),
            ("model", LinearRegression()),
        ]
    )
    reg_pipeline.fit(Xr_train, yr_train)
    yr_pred = reg_pipeline.predict(Xr_test)

    rmse = float(np.sqrt(mean_squared_error(yr_test, yr_pred)))
    mae = float(mean_absolute_error(yr_test, yr_pred))
    r2 = float(r2_score(yr_test, yr_pred))
    residual_std = float(np.std(yr_test - yr_pred))

    _save_residual_plot(yr_test.to_numpy(), yr_pred, results_dir / "residual_plot.png")

    metrics = {
        "classification_accuracy": float(cls_accuracy),
        "classification_precision_weighted": float(cls_precision),
        "classification_recall_weighted": float(cls_recall),
        "clustering_best_k": int(best_k),
        "clustering_silhouette": float(best_score),
        "regression_rmse": rmse,
        "regression_mae": mae,
        "regression_r2": r2,
        "regression_residual_std": residual_std,
    }

    # Serialize artifacts
    joblib.dump(dt_pipeline, models_dir / "decision_tree_classifier.joblib")
    joblib.dump(cluster_scaler, models_dir / "cluster_scaler.joblib")
    joblib.dump(best_model, models_dir / "kmeans_cluster_model.joblib")
    joblib.dump(knn_cluster_inference, models_dir / "knn_cluster_inference.joblib")
    joblib.dump(reg_pipeline, models_dir / "linear_regression_model.joblib")

    artifacts = {
        "cluster_feature_order": cluster_features,
        "classification_feature_order": [
            "Area",
            "Year",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp",
        ],
        "regression_feature_order": [
            "Area",
            "Item",
            "Year",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp",
        ],
        "cluster_guidance": cluster_guidance,
        "metrics": metrics,
    }

    (models_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    metrics = train_all(
        data_path=project_root / "data" / "yield_df.csv",
        models_dir=project_root / "models",
        results_dir=project_root / "results",
    )
    print(json.dumps(metrics, indent=2))
