# payorlens/risk_interpreter.py
"""
PayorLens RiskInterpreter — The 'So What' Layer
Architecture v2.0

Translates raw metric numbers into:
  1. Buyer-facing payor risk narrative (what a compliance officer reads)
  2. NIST AI RMF function + subcategory mapping
  3. Recommended action
  4. Risk level: LOW | MEDIUM | HIGH | CRITICAL

This module is the commercial differentiator. No fairness notebook on GitHub
has this. Every metric output from evaluator.py, fairness.py, and robustness.py
passes through here before reaching the report.
"""

from dataclasses import dataclass, field
from typing import Literal, List, Optional

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

RISK_COLORS = {
    "LOW":      "#1E7A4A",
    "MEDIUM":   "#C8932A",
    "HIGH":     "#E67E22",
    "CRITICAL": "#B03A2E",
}

RISK_BADGES = {
    "LOW":      "🟢 LOW",
    "MEDIUM":   "🟡 MEDIUM",
    "HIGH":     "🟠 HIGH",
    "CRITICAL": "🔴 CRITICAL",
}


# ── NIST AI RMF reference table ───────────────────────────────────────────────
NIST_MAP = {
    "fairness_dpd"     : ("Measure",  "MS-2.5",  "Bias and fairness testing across demographic cohorts"),
    "fairness_eod"     : ("Measure",  "MS-2.5",  "Equalized odds and error rate disparity"),
    "calibration"      : ("Measure",  "MS-2.3",  "AI output reliability and uncertainty quantification"),
    "high_conf_error"  : ("Manage",   "MG-2.4",  "High-consequence error routing and human oversight"),
    "robustness"       : ("Measure",  "MS-2.6",  "Robustness and resilience under input perturbation"),
    "data_quality"     : ("Map",      "MP-2.3",  "Data provenance, quality, and lineage"),
    "overall"          : ("Govern",   "GV-1.1",  "Organisational AI risk tolerance and governance posture"),
}


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class RiskFinding:
    metric_name         : str
    metric_value        : float
    risk_level          : RiskLevel
    nist_function       : str
    nist_subcategory    : str
    nist_description    : str
    payor_interpretation: str
    recommended_action  : str
    statistical_note    : str = ""

    def badge(self) -> str:
        return RISK_BADGES[self.risk_level]

    def color(self) -> str:
        return RISK_COLORS[self.risk_level]


@dataclass
class ExecutiveSummary:
    overall_risk        : Literal["GREEN", "AMBER", "RED"]
    total_findings      : int
    critical_count      : int
    high_count          : int
    medium_count        : int
    low_count           : int
    recommendation      : str
    top_findings        : List[str]
    model_name          : str
    dataset_description : str = "CMS DE-SynPUF Inpatient Claims"


# ── RiskInterpreter ────────────────────────────────────────────────────────────
class RiskInterpreter:

    # ── DPD (Demographic Parity Difference) ───────────────────────────────────
    def interpret_dpd(self, dpd: float, feature: str,
                      p_value: float, model_name: str) -> RiskFinding:
        nist = NIST_MAP["fairness_dpd"]
        sig  = p_value < 0.05

        if not sig:
            risk   = "LOW"
            interp = (f"DPD of {dpd:.3f} across {feature} cohorts is NOT statistically "
                      f"significant (p={p_value:.3f}). No actionable bias finding at this time.")
            action = "Continue monitoring. Schedule retest after model update or data refresh."

        elif dpd < 0.05:
            risk   = "LOW"
            interp = (f"Statistically significant but small denial rate disparity ({dpd:.3f}) "
                      f"across {feature} cohorts (p={p_value:.4f}). Within acceptable tolerance.")
            action = "Document the finding. Flag for re-evaluation if DPD increases post-retraining."

        elif dpd < 0.10:
            risk   = "MEDIUM"
            interp = (f"Meaningful denial rate disparity ({dpd:.3f}) across {feature} cohorts "
                      f"(p={p_value:.4f}). Under NIST AI RMF MS-2.5, this constitutes a measurable "
                      f"fairness gap. In a prior auth workflow, this pattern would attract NAIC "
                      f"unfair discrimination scrutiny and requires documented mitigation.")
            action = ("Investigate root cause in training data. Rebalance cohort representation "
                      "or apply post-processing fairness constraint. Document remediation steps.")

        elif dpd < 0.15:
            risk   = "HIGH"
            interp = (f"High denial rate disparity ({dpd:.3f}) across {feature} cohorts "
                      f"(p={p_value:.4f}). Exceeds NIST AI RMF tolerance for high-stakes AI. "
                      f"A payer deploying this model in coverage decisions risks regulatory "
                      f"examination under Colorado SB21-169 and NY Circular Letter No. 7.")
            action = ("Escalate to model owner and compliance lead. Suspend deployment of "
                      "affected cohort logic until retrained. Produce written remediation plan.")

        else:
            risk   = "CRITICAL"
            interp = (f"Severe denial rate disparity (DPD={dpd:.3f}) across {feature} cohorts "
                      f"(p={p_value:.4f}). This surpasses the NAIC Adverse Impact Ratio threshold "
                      f"implication (0.80–1.25 bounds). A payer using this model in prior auth or "
                      f"claims adjudication faces litigation exposure comparable to the UnitedHealth "
                      f"nH Predict and Cigna PxDx class-action pattern.")
            action = ("Do NOT deploy in production. Full model audit required. "
                      "Independent third-party re-validation recommended before any live use.")

        return RiskFinding(
            metric_name=f"DPD ({feature})", metric_value=dpd,
            risk_level=risk, nist_function=nist[0], nist_subcategory=nist[1],
            nist_description=nist[2], payor_interpretation=interp,
            recommended_action=action,
            statistical_note=f"chi-square p-value={p_value:.4f}, significant={sig}"
        )

    # ── Calibration / Brier score ─────────────────────────────────────────────
    def interpret_calibration(self, brier_score: float, high_conf_wrong: int,
                               total_samples: int) -> RiskFinding:
        nist = NIST_MAP["calibration"]
        hce_rate = high_conf_wrong / max(total_samples, 1)

        if brier_score < 0.05 and hce_rate < 0.02:
            risk   = "LOW"
            interp = (f"Model is well-calibrated (Brier={brier_score:.3f}). "
                      f"High-confidence error rate is {hce_rate*100:.1f}% — acceptable.")
            action = "Maintain current calibration. Re-check after any retraining."

        elif brier_score < 0.15:
            risk   = "MEDIUM"
            interp = (f"Moderate calibration gap (Brier={brier_score:.3f}). "
                      f"{high_conf_wrong} predictions had >85% model confidence but were wrong "
                      f"({hce_rate*100:.1f}% of test set). Confidence scores should not be "
                      f"used as sole basis for automated approval routing.")
            action = "Apply Platt scaling or isotonic regression calibration. Retest before deployment."

        elif hce_rate > 0.05:
            risk   = "HIGH"
            interp = (f"Poor calibration (Brier={brier_score:.3f}) with {high_conf_wrong} "
                      f"high-confidence wrong predictions ({hce_rate*100:.1f}%). "
                      f"In prior auth AI, high-confidence wrong denials bypass human review "
                      f"and become adverse determinations — the exact failure mode cited in "
                      f"the Cigna PxDx litigation (avg 1.2 seconds per claim review).")
            action = ("Implement confidence threshold gating: require human review for all "
                      "decisions where model confidence >85% until calibration is corrected.")

        else:
            risk   = "CRITICAL"
            interp = (f"Severely miscalibrated model (Brier={brier_score:.3f}). "
                      f"Confidence scores are unreliable. Autonomous use in coverage "
                      f"decisions is unjustifiable under CMS-0057-F explainability requirements.")
            action = "Do not use model confidence for routing logic. Full recalibration required."

        return RiskFinding(
            metric_name="Calibration (Brier Score + High-Conf Errors)",
            metric_value=brier_score, risk_level=risk,
            nist_function=nist[0], nist_subcategory=nist[1], nist_description=nist[2],
            payor_interpretation=interp, recommended_action=action,
            statistical_note=f"High-confidence errors (>85% conf, wrong): {high_conf_wrong}/{total_samples}"
        )

    # ── Robustness degradation ────────────────────────────────────────────────
    def interpret_robustness(self, scenario_name: str, baseline_f1: float,
                              degraded_f1: float, null_rate: float) -> RiskFinding:
        nist      = NIST_MAP["robustness"]
        decay_pct = (baseline_f1 - degraded_f1) / max(baseline_f1, 0.001) * 100

        if decay_pct <= 5:
            risk   = "LOW"
            interp = (f"Model shows negligible performance degradation ({decay_pct:.1f}% F1 decay) "
                      f"under '{scenario_name}' at {null_rate*100:.0f}% corruption rate. "
                      f"Robust to this failure mode under NIST AI RMF MS-2.6.")
            action = "No action required. Document robustness test result for governance trail."

        elif decay_pct <= 10:
            risk   = "MEDIUM"
            interp = (f"Moderate F1 degradation ({decay_pct:.1f}%) under '{scenario_name}' "
                      f"at {null_rate*100:.0f}% corruption. Crosses warning threshold (>5%). "
                      f"In real payer workflows, data quality issues at this rate are common "
                      f"(incomplete PA submissions, missing documentation). Performance will "
                      f"degrade in production without pipeline quality controls.")
            action = "Implement upstream data validation. Add fallback to human review when key fields are missing."

        elif decay_pct <= 20:
            risk   = "HIGH"
            interp = (f"High F1 degradation ({decay_pct:.1f}%) under '{scenario_name}' "
                      f"at {null_rate*100:.0f}% corruption. Crosses warning threshold (>10%). "
                      f"This simulates a clinically plausible payer data pipeline failure. "
                      f"A model this fragile must not be deployed without guaranteed upstream "
                      f"data quality controls per NIST AI RMF MS-2.6.")
            action = ("Establish data quality SLA upstream of model. "
                      "Auto-route claims with missing required fields to human review, not model.")

        else:
            risk   = "CRITICAL"
            interp = (f"Severe F1 collapse ({decay_pct:.1f}%) under '{scenario_name}'. "
                      f"Crosses DANGER threshold (>20%). Model is critically dependent on "
                      f"data completeness. Deployment in a live payer workflow without guaranteed "
                      f"data quality would produce systematically wrong decisions — a governance "
                      f"failure under NIST AI RMF MS-2.6.")
            action = ("BLOCK deployment until data quality guarantee is contractually enforced. "
                      "Retrain with explicit missing-value robustness augmentation.")

        return RiskFinding(
            metric_name=f"Robustness — {scenario_name}",
            metric_value=round(decay_pct, 2), risk_level=risk,
            nist_function=nist[0], nist_subcategory=nist[1], nist_description=nist[2],
            payor_interpretation=interp, recommended_action=action,
            statistical_note=(
                f"Baseline F1={baseline_f1:.3f} → Degraded F1={degraded_f1:.3f} "
                f"(decay={decay_pct:.1f}%) | Warning >5% | Danger >20%"
            )
        )

    # ── Data quality ──────────────────────────────────────────────────────────
    def interpret_data_quality(self, error_rate: float, total_records: int) -> RiskFinding:
        nist = NIST_MAP["data_quality"]
        if error_rate < 0.01:
            risk   = "LOW"
            interp = (f"Data validation error rate of {error_rate*100:.2f}% — excellent. "
                      f"Pydantic schema enforcement is effective on this dataset.")
            action = "Maintain current data contract. Re-validate on any schema change."
        elif error_rate < 0.05:
            risk   = "MEDIUM"
            interp = (f"Validation error rate of {error_rate*100:.2f}% ({int(total_records*error_rate):,} records). "
                      f"Indicates upstream data quality issues that would accumulate at scale.")
            action = "Investigate error patterns. Add pre-validation data cleaning step."
        else:
            risk   = "HIGH"
            interp = (f"High validation error rate of {error_rate*100:.2f}%. "
                      f"This volume of malformed records in production would produce "
                      f"unreliable model inputs and governance gaps in the audit trail.")
            action = "Mandatory data quality remediation before any model training or evaluation."

        return RiskFinding(
            metric_name="Data Quality (Validation Error Rate)",
            metric_value=error_rate, risk_level=risk,
            nist_function=nist[0], nist_subcategory=nist[1], nist_description=nist[2],
            payor_interpretation=interp, recommended_action=action,
            statistical_note=f"Total records evaluated: {total_records:,}"
        )

    # ── Executive summary ─────────────────────────────────────────────────────
    def generate_executive_summary(self, findings: List[RiskFinding],
                                    model_name: str) -> ExecutiveSummary:
        critical = [f for f in findings if f.risk_level == "CRITICAL"]
        high     = [f for f in findings if f.risk_level == "HIGH"]
        medium   = [f for f in findings if f.risk_level == "MEDIUM"]
        low      = [f for f in findings if f.risk_level == "LOW"]

        if critical:
            overall = "RED"
            rec = (f"DO NOT DEPLOY. {len(critical)} critical finding(s) must be remediated "
                   "before this model is used in any coverage decision workflow.")
        elif high:
            overall = "AMBER"
            rec = (f"REMEDIATE BEFORE PRODUCTION. {len(high)} high-risk finding(s) identified. "
                   "Deployment in coverage decisions requires documented mitigation.")
        else:
            overall = "GREEN"
            rec = ("Model meets baseline governance standards for evaluation purposes. "
                   "Monitor continuously after any retraining or data distribution change.")

        top_findings = [f.payor_interpretation[:200] + "…"
                        for f in (critical + high + medium)[:3]]

        return ExecutiveSummary(
            overall_risk=overall,
            total_findings=len(findings),
            critical_count=len(critical),
            high_count=len(high),
            medium_count=len(medium),
            low_count=len(low),
            recommendation=rec,
            top_findings=top_findings,
            model_name=model_name,
        )