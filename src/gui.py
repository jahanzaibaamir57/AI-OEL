import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import InferenceEngine


class SmartAgriApp:
    def __init__(self, root: tk.Tk, project_root: Path) -> None:
        self.root = root
        self.project_root = project_root
        self.root.title("Smart Agriculture Decision Support System")
        self.root.geometry("1280x760")

        self.engine = InferenceEngine(project_root / "models")
        self.metrics = json.loads((project_root / "results" / "metrics.json").read_text(encoding="utf-8"))

        self._build_layout()

    def _build_layout(self) -> None:
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(main_frame, text="Input Parameters", padding=12)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(main_frame)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Inputs
        self.area_var = tk.StringVar(value="Albania")
        self.year_var = tk.StringVar(value="2013")
        self.rain_var = tk.StringVar(value="1200")
        self.pesticide_var = tk.StringVar(value="110")
        self.temp_var = tk.StringVar(value="21")

        self._input_row(left, "Area", self.area_var)
        self._input_row(left, "Year", self.year_var)
        self._input_row(left, "Average Rainfall (mm/year)", self.rain_var)
        self._input_row(left, "Pesticides (tonnes)", self.pesticide_var)
        self._input_row(left, "Average Temperature", self.temp_var)

        ttk.Button(left, text="Run Integrated Inference", command=self.run_inference).pack(pady=10, fill=tk.X)

        self.output_box = tk.Text(left, width=44, height=20, wrap=tk.WORD)
        self.output_box.pack(fill=tk.BOTH, expand=True, pady=8)

        self._build_plot_panel(right)

    def _input_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text=label, width=28).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=20).pack(side=tk.RIGHT)

    def _build_plot_panel(self, parent: ttk.Frame) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

        image_paths = [
            self.project_root / "results" / "feature_importance.png",
            self.project_root / "results" / "cluster_scatter.png",
            self.project_root / "results" / "residual_plot.png",
        ]
        titles = ["Feature Importance", "Cluster Distribution", "Residual Analysis"]

        for ax, image_path, title in zip(axes, image_paths, titles):
            if image_path.exists():
                img = mpimg.imread(image_path)
                ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_inference(self) -> None:
        try:
            user_input = {
                "Area": self.area_var.get().strip(),
                "Year": int(self.year_var.get()),
                "average_rain_fall_mm_per_year": float(self.rain_var.get()),
                "pesticides_tonnes": float(self.pesticide_var.get()),
                "avg_temp": float(self.temp_var.get()),
            }
            result = self.engine.predict(user_input)
        except Exception as exc:
            messagebox.showerror("Input/Inference Error", str(exc))
            return

        self.output_box.delete("1.0", tk.END)
        self.output_box.insert(
            tk.END,
            "Integrated AI Output\n"
            "---------------------\n"
            f"Recommended Crop: {result['recommended_crop']}\n"
            f"Assigned Soil Zone Cluster: {result['cluster_id']}\n"
            f"Agronomic Guidance: {result['cluster_guidance']}\n"
            f"Predicted Yield (hg/ha): {result['predicted_yield']}\n"
            f"95% Confidence Bounds: {result['yield_confidence']}\n\n"
            "Key Model Metrics\n"
            "------------------\n"
            f"Decision Tree Accuracy: {self.metrics.get('classification_accuracy', 0):.4f}\n"
            f"Decision Tree Precision: {self.metrics.get('classification_precision_weighted', 0):.4f}\n"
            f"Decision Tree Recall: {self.metrics.get('classification_recall_weighted', 0):.4f}\n"
            f"Clustering Silhouette: {self.metrics.get('clustering_silhouette', 0):.4f}\n"
            f"Regression RMSE: {self.metrics.get('regression_rmse', 0):.2f}\n"
            f"Regression MAE: {self.metrics.get('regression_mae', 0):.2f}\n"
            f"Regression R2: {self.metrics.get('regression_r2', 0):.4f}\n"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    root = tk.Tk()
    SmartAgriApp(root, project_root)
    root.mainloop()


if __name__ == "__main__":
    main()
