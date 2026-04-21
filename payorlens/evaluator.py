# payorlens/evaluator.py
"""
PayorLens EvalEngine — Model Training & Core Performance Metrics

"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute        import SimpleImputer
from sklearn.compose       import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration   import calibration_curve, CalibratedClassifierCV
from sklearn.metrics       import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
    confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger("PayorLens.Evaluator")

# ── Feature definition ────────────────────────────────────────────────────────
# claim_amount EXCLUDED: it was the denial_status proxy in v1 (CLM_PMT_AMT==0),
# keeping it in features caused target leakage → AUC=1.0. Removed permanently.
NUMERIC_FEATURES = [
    "age", "utilization_days", "deductible_amount", "chronic_count",
    "has_diabetes", "has_chf", "has_copd", "has_cancer",
]
CATEGORICAL_FEATURES = ["gender", "race", "age_band", "state_code"]
TARGET = "denial_status"


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


def build_models(preprocessor: ColumnTransformer) -> dict:
    return {
        "logistic": Pipeline([
            ("pre",   preprocessor),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            )),
        ]),
        "gbm": Pipeline([
            ("pre",   preprocessor),
            ("model", GradientBoostingClassifier(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=42
            )),
        ]),
    }


# ── EvalEngine ────────────────────────────────────────────────────────────────
class EvalEngine:

    def __init__(self, output_dir: str = "D:/payorlens/data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict = {}
        self.models: dict  = {}

    def load_data(self, parquet_path: str):
        logger.info(f"Loading data from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        X  = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y  = df[TARGET]
        logger.info(f"Dataset: {len(df):,} rows  |  denial rate: {y.mean()*100:.1f}%")
        return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    def train_and_evaluate(self, parquet_path: str) -> dict:
        X_train, X_test, y_train, y_test = self.load_data(parquet_path)

        preprocessor = build_preprocessor()
        model_specs   = build_models(preprocessor)

        # Pre-compute balanced sample weights for GBM
        # (GradientBoosting has no class_weight param — must pass sample_weight)
        from sklearn.utils.class_weight import compute_sample_weight
        balanced_sw = compute_sample_weight("balanced", y_train)

        for name, pipeline in model_specs.items():
            logger.info(f"Training {name} …")
            if name == "gbm":
                # Pass sample_weight through Pipeline: key = "model__sample_weight"
                pipeline.fit(X_train, y_train, model__sample_weight=balanced_sw)
            else:
                pipeline.fit(X_train, y_train)
            self.models[name] = pipeline

            y_pred  = pipeline.predict(X_test)
            y_prob  = pipeline.predict_proba(X_test)[:, 1]

            metrics = self._compute_metrics(y_test, y_pred, y_prob, name)
            self.results[name] = metrics
            logger.info(f"{name}  F1={metrics['f1']:.3f}  AUC={metrics['roc_auc']:.3f}  "
                        f"Brier={metrics['brier_score']:.3f}")

        # Save models + charts
        self._save_models()
        self._plot_roc(y_test)
        self._plot_calibration(y_test)

        return self.results

    # ── metrics ───────────────────────────────────────────────────────────────
    def _compute_metrics(self, y_true, y_pred, y_prob, name: str) -> dict:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        cm = confusion_matrix(y_true, y_pred)
        high_conf_wrong = self._high_confidence_errors(y_true, y_pred, y_prob)

        return {
            "model_name"            : name,
            "accuracy"              : round(float(accuracy_score(y_true, y_pred)), 4),
            "f1"                    : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc"               : round(float(roc_auc_score(y_true, y_prob)), 4),
            "brier_score"           : round(float(brier_score_loss(y_true, y_prob)), 4),
            "confusion_matrix"      : cm.tolist(),
            "calibration_frac_pos"  : frac_pos.tolist(),
            "calibration_mean_pred" : mean_pred.tolist(),
            "high_conf_wrong_count" : high_conf_wrong["count"],
            "high_conf_wrong_rate"  : round(high_conf_wrong["rate"], 4),
            "high_conf_threshold"   : 0.85,
            "total_test_samples"    : len(y_true),
            # stored for failure narrative extraction in cli.py
            "_y_true"               : list(y_true),
            "_y_pred"               : list(y_pred),
            "_y_prob"               : list(y_prob),
        }

    def _high_confidence_errors(self, y_true, y_pred, y_prob, threshold=0.85) -> dict:
        """
        The 'dangerous prediction' metric.
        Model was >85% confident AND wrong — the exact failure mode
        cited in UnitedHealth / Cigna litigation (high confidence + wrong denial).
        """
        mask  = (y_prob > threshold) & (np.array(y_pred) != np.array(y_true))
        count = int(mask.sum())
        rate  = count / max(len(y_true), 1)
        return {"count": count, "rate": rate}

    # ── persistence ───────────────────────────────────────────────────────────
    def _save_models(self):
        for name, model in self.models.items():
            path = self.output_dir / f"model_{name}.pkl"
            joblib.dump(model, path)
            logger.info(f"Model saved → {path}")

    # ── charts ────────────────────────────────────────────────────────────────
    def _plot_roc(self, y_test):
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = {"logistic": "#0A7EA4", "gbm": "#C8932A"}
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
        for name, res in self.results.items():
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(res["_y_true"], res["_y_prob"])
            auc = res["roc_auc"]
            ax.plot(fpr, tpr, color=colors.get(name, "#333"),
                    linewidth=2, label=f"{name}  (AUC={auc:.3f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — PayorLens Model Comparison",
                      fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        path = self.output_dir / "chart_roc.png"
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        logger.info(f"ROC chart → {path}")

    def _plot_calibration(self, y_test):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
        colors = {"logistic": "#0A7EA4", "gbm": "#C8932A"}
        for name, res in self.results.items():
            ax.plot(res["calibration_mean_pred"], res["calibration_frac_pos"],
                    "s-", color=colors.get(name, "#333"), label=name)

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve — Reliability Diagram", fontsize=13, fontweight="bold")
        ax.legend()
        path = self.output_dir / "chart_calibration.png"
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        logger.info(f"Calibration chart → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    PARQUET = "D:/payorlens/data/processed/claims_v1.parquet"
    OUT_DIR  = "D:/payorlens/data/processed"

    engine  = EvalEngine(output_dir=OUT_DIR)
    results = engine.train_and_evaluate(PARQUET)

    print("\n--- Day 3 & 4 Milestone: Model Evaluation Complete ---")
    for name, r in results.items():
        print(f"\n[{name.upper()}]")
        print(f"  Accuracy        : {r['accuracy']:.4f}")
        print(f"  F1 Score        : {r['f1']:.4f}")
        print(f"  ROC-AUC         : {r['roc_auc']:.4f}")
        print(f"  Brier Score     : {r['brier_score']:.4f}  (lower = better calibration)")
        print(f"  High-Conf Wrong : {r['high_conf_wrong_count']} "
              f"({r['high_conf_wrong_rate']*100:.2f}% of test set)")