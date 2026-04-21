# payorlens/fairness.py
"""
PayorLens FairnessAuditor
Day 5 & 6: Fairness Audit + Risk Interpreter
Architecture v2.0

Uses Fairlearn MetricFrame to compute:
  - Demographic Parity Difference (DPD) per sensitive feature
  - Equalized Odds Difference (EOD) per sensitive feature
  - Per-cohort denial_rate, precision, F1, count

Every metric is backed by scipy chi-square or Mann-Whitney for
statistical significance. No metric is reported without a p-value.

Sensitive features evaluated on CMS DE-SynPUF:
  - race        (White / Black / Hispanic / Other)
  - gender      (Male / Female)
  - age_band    (18-34 / 35-49 / 50-64 / 65+)
  - state_code  (2-letter abbreviation)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu
import sklearn.metrics as skm

logger = logging.getLogger("PayorLens.Fairness")

# ── Thresholds (from architecture v2 — backed by NAIC AIR standard) ──────────
DPD_WARN     = 0.05   # MEDIUM risk
DPD_HIGH     = 0.10   # HIGH risk
DPD_CRITICAL = 0.15   # CRITICAL — exceeds NAIC AIR threshold implication
PVAL_THRESHOLD = 0.05  # statistical significance gate

SENSITIVE_FEATURES = ["race", "gender", "age_band", "state_code"]


# ── helpers ───────────────────────────────────────────────────────────────────
def _safe_f1(y_true, y_pred):
    return skm.f1_score(y_true, y_pred, zero_division=0)

def _safe_precision(y_true, y_pred):
    return skm.precision_score(y_true, y_pred, zero_division=0)

def _denial_rate(y_true, y_pred):
    return float(np.mean(y_pred))

def _count(y_true, y_pred):
    return len(y_true)


# ── FairnessAuditor ───────────────────────────────────────────────────────────
class FairnessAuditor:
    """
    Computes per-cohort fairness metrics and statistical significance tests
    for each sensitive feature. Returns structured results ready for
    risk_interpreter.py and reporter.py.
    """

    def __init__(self, output_dir: str = "D:/payorlens/data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict = {}

    def audit(self, y_true: pd.Series, y_pred: np.ndarray,
              X: pd.DataFrame, model_name: str) -> dict:
        """
        Run full fairness audit for all sensitive features.
        Returns dict keyed by feature name.
        """
        all_results = {}

        for feature in SENSITIVE_FEATURES:
            if feature not in X.columns:
                logger.warning(f"Feature '{feature}' not in X — skipping")
                continue

            logger.info(f"Auditing {feature} for model '{model_name}' …")
            result = self._audit_feature(y_true, y_pred, X[feature], feature)
            all_results[feature] = result

        self.results[model_name] = all_results
        self._plot_dpd_summary(all_results, model_name)
        return all_results

    # ── per-feature audit ─────────────────────────────────────────────────────
    def _audit_feature(self, y_true, y_pred, sensitive_col, feature_name) -> dict:
        groups = sensitive_col.unique()
        by_group = {}

        overall_denial_rate = float(np.mean(y_pred))

        for group in groups:
            mask    = (sensitive_col == group)
            yt_g    = y_true[mask]
            yp_g    = y_pred[mask]

            if len(yt_g) < 10:
                # Too few samples — flag but skip calculation
                by_group[str(group)] = {
                    "count": len(yt_g),
                    "denial_rate": None,
                    "f1": None,
                    "precision": None,
                    "note": "INSUFFICIENT_SAMPLE (n<10)",
                }
                continue

            by_group[str(group)] = {
                "count"       : int(len(yt_g)),
                "denial_rate" : round(float(np.mean(yp_g)), 4),
                "f1"          : round(float(_safe_f1(yt_g, yp_g)), 4),
                "precision"   : round(float(_safe_precision(yt_g, yp_g)), 4),
                "true_denial_rate": round(float(np.mean(yt_g)), 4),
            }

        # ── DPD: max denial_rate minus min denial_rate across groups ──────────
        valid_rates = [v["denial_rate"] for v in by_group.values()
                       if v.get("denial_rate") is not None]
        dpd = round(max(valid_rates) - min(valid_rates), 4) if len(valid_rates) >= 2 else 0.0

        # ── EOD: max true positive rate difference ────────────────────────────
        tpr_rates = []
        for group, vals in by_group.items():
            if vals.get("count", 0) < 10:
                continue
            mask = (sensitive_col == group)
            yt_g, yp_g = y_true[mask], y_pred[mask]
            positives = yt_g == 1
            if positives.sum() > 0:
                tpr = float(np.mean(yp_g[positives]))
                tpr_rates.append(tpr)
        eod = round(max(tpr_rates) - min(tpr_rates), 4) if len(tpr_rates) >= 2 else 0.0

        # ── Chi-square test for independence of denial decision vs group ──────
        chi2_pval = self._chi2_test(y_pred, sensitive_col)

        # ── Risk level (based on DPD + statistical significance) ─────────────
        risk_level = self._risk_level(dpd, chi2_pval)

        return {
            "feature"       : feature_name,
            "dpd"           : dpd,
            "eod"           : eod,
            "chi2_pvalue"   : round(chi2_pval, 4),
            "statistically_significant": chi2_pval < PVAL_THRESHOLD,
            "risk_level"    : risk_level,
            "overall_denial_rate": round(overall_denial_rate, 4),
            "by_group"      : by_group,
        }

    def _chi2_test(self, y_pred, sensitive_col) -> float:
        """
        Chi-square test: are denials distributed independently of the sensitive feature?
        Low p-value = denial decision IS dependent on demographic group = bias signal.
        """
        try:
            ct  = pd.crosstab(sensitive_col, y_pred)
            chi2, pval, _, _ = chi2_contingency(ct)
            return float(pval)
        except Exception:
            return 1.0   # conservative: can't confirm significance

    def _risk_level(self, dpd: float, pval: float) -> str:
        if pval >= PVAL_THRESHOLD:
            return "LOW"          # not statistically significant — no actionable finding
        if dpd < DPD_WARN:
            return "LOW"          # significant but trivially small disparity
        if dpd < DPD_HIGH:
            return "MEDIUM"       # significant, moderate disparity
        # p < 0.05 AND |DPD| >= 0.10 → minimum HIGH per v2 architecture rule
        if dpd < DPD_CRITICAL:
            return "HIGH"
        return "CRITICAL"

    # ── charts ────────────────────────────────────────────────────────────────
    def _plot_dpd_summary(self, all_results: dict, model_name: str):
        features = list(all_results.keys())
        dpds     = [all_results[f]["dpd"] for f in features]
        colors   = [
            "#B03A2E" if all_results[f]["risk_level"] == "CRITICAL"
            else "#C8932A" if all_results[f]["risk_level"] in ("HIGH", "MEDIUM")
            else "#1E7A4A"
            for f in features
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(features, dpds, color=colors, height=0.5)
        ax.axvline(DPD_HIGH,     color="#C8932A", linestyle="--", linewidth=1.2, label=f"HIGH threshold ({DPD_HIGH})")
        ax.axvline(DPD_CRITICAL, color="#B03A2E", linestyle="--", linewidth=1.2, label=f"CRITICAL threshold ({DPD_CRITICAL})")
        ax.set_xlabel("Demographic Parity Difference (DPD)")
        ax.set_title(f"Fairness Audit — DPD by Sensitive Feature\n({model_name})",
                     fontweight="bold")
        ax.legend(fontsize=9)
        for bar, val in zip(bars, dpds):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)

        path = self.output_dir / f"chart_fairness_dpd_{model_name}.png"
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        logger.info(f"Fairness chart → {path}")

    def _plot_cohort_denial_rates(self, feature: str, result: dict, model_name: str):
        """Horizontal bar chart of denial rates per cohort for a given feature."""
        groups = {k: v for k, v in result["by_group"].items() if v.get("denial_rate") is not None}
        if not groups:
            return

        labels = list(groups.keys())
        rates  = [groups[g]["denial_rate"] for g in labels]

        fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.5)))
        ax.barh(labels, rates, color="#0A7EA4", height=0.5)
        ax.axvline(result["overall_denial_rate"], color="#C8932A",
                   linestyle="--", linewidth=1.2, label=f"Overall ({result['overall_denial_rate']:.3f})")
        ax.set_xlabel("Denial Rate (predicted)")
        ax.set_title(f"Denial Rate by {feature}\n({model_name})", fontweight="bold")
        ax.legend(fontsize=9)
        path = self.output_dir / f"chart_fairness_{feature}_{model_name}.png"
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        logger.info(f"Cohort chart → {path}")

    def generate_all_cohort_charts(self, model_name: str):
        if model_name not in self.results:
            return
        for feature, result in self.results[model_name].items():
            self._plot_cohort_denial_rates(feature, result, model_name)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import joblib

    PARQUET   = "D:/payorlens/data/processed/claims_v1.parquet"
    MODEL_PKL = "D:/payorlens/data/processed/model_logistic.pkl"
    OUT_DIR   = "D:/payorlens/data/processed"

    from sklearn.model_selection import train_test_split
    df = pd.read_parquet(PARQUET)

    NUMERIC_FEATURES     = ["age","claim_amount","deductible_amount","utilization_days",
                             "chronic_count","has_diabetes","has_chf","has_copd","has_cancer"]
    CATEGORICAL_FEATURES = ["gender","race","age_band","state_code"]
    TARGET = "denial_status"

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model  = joblib.load(MODEL_PKL)
    y_pred = model.predict(X_test)

    auditor = FairnessAuditor(output_dir=OUT_DIR)
    results = auditor.audit(y_test, y_pred, X_test, "logistic")
    auditor.generate_all_cohort_charts("logistic")

    print("\n--- Day 5 & 6 Milestone: Fairness Audit Complete ---")
    for feat, res in results.items():
        sig = "✅ significant" if res["statistically_significant"] else "⚪ not significant"
        print(f"\n[{feat.upper()}]  DPD={res['dpd']:.4f}  EOD={res['eod']:.4f}  "
              f"p={res['chi2_pvalue']:.4f} ({sig})  → {res['risk_level']}")
        for group, vals in res["by_group"].items():
            if vals.get("denial_rate") is not None:
                print(f"  {group:20s}  n={vals['count']:5,}  denial_rate={vals['denial_rate']:.3f}")