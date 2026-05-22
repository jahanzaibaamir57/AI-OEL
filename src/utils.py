import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd


class InferenceEngine:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.cls_model = joblib.load(models_dir / "decision_tree_classifier.joblib")
        self.cluster_scaler = joblib.load(models_dir / "cluster_scaler.joblib")
        self.cluster_knn = joblib.load(models_dir / "knn_cluster_inference.joblib")
        self.reg_model = joblib.load(models_dir / "linear_regression_model.joblib")

        self.artifacts = json.loads((models_dir / "artifacts.json").read_text(encoding="utf-8"))
        self.metrics = self.artifacts.get("metrics", {})

    def predict(self, user_input: Dict[str, float]) -> Dict[str, str]:
        cls_features = pd.DataFrame(
            [
                {
                    "Area": user_input["Area"],
                    "Year": user_input["Year"],
                    "average_rain_fall_mm_per_year": user_input["average_rain_fall_mm_per_year"],
                    "pesticides_tonnes": user_input["pesticides_tonnes"],
                    "avg_temp": user_input["avg_temp"],
                }
            ]
        )

        recommended_crop = str(self.cls_model.predict(cls_features)[0])

        cluster_vector = np.array(
            [
                [
                    user_input["average_rain_fall_mm_per_year"],
                    user_input["pesticides_tonnes"],
                    user_input["avg_temp"],
                ]
            ]
        )
        cluster_scaled = self.cluster_scaler.transform(cluster_vector)
        cluster_id = int(self.cluster_knn.predict(cluster_scaled)[0])

        reg_features = pd.DataFrame(
            [
                {
                    "Area": user_input["Area"],
                    "Item": recommended_crop,
                    "Year": user_input["Year"],
                    "average_rain_fall_mm_per_year": user_input["average_rain_fall_mm_per_year"],
                    "pesticides_tonnes": user_input["pesticides_tonnes"],
                    "avg_temp": user_input["avg_temp"],
                }
            ]
        )

        predicted_yield = float(self.reg_model.predict(reg_features)[0])
        residual_std = float(self.metrics.get("regression_residual_std", 0.0))
        margin = 1.96 * residual_std

        guidance_map = self.artifacts.get("cluster_guidance", {})
        guidance = guidance_map.get(
            str(cluster_id),
            "General guidance: maintain balanced fertilization and water scheduling.",
        )

        return {
            "recommended_crop": recommended_crop,
            "cluster_id": str(cluster_id),
            "cluster_guidance": guidance,
            "predicted_yield": f"{predicted_yield:.2f}",
            "yield_confidence": f"[{predicted_yield - margin:.2f}, {predicted_yield + margin:.2f}]",
        }
