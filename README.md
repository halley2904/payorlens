# PayorLens

**AI governance evaluation harness for clinical decision models in insurance workflows.**

Health insurers are deploying AI in prior authorization and claims adjudication. Most teams can run a fairness notebook. Almost none can explain what a DPD of 0.286 across race cohorts means for their regulatory exposure, or what a Brier score of 0.236 means for their CMS-0057-F explainability obligations.

PayorLens bridges the gap between raw evaluation metrics and the business consequence — producing a two-audience governance report that a compliance officer and an ML engineer can both act on.

---

## Sample findings

> Evaluated on **CMS DE-SynPUF** inpatient claims (66,718 records) · Logistic Regression baseline model

| Finding | Metric | Risk Level |
|---------|--------|------------|
| Model calibration failure | Brier = 0.236 · 8 high-confidence wrong predictions | 🔴 CRITICAL |
| Race cohort denial disparity | DPD = 0.286 · p < 0.0001 | 🔴 CRITICAL |
| Age band denial disparity | DPD = 0.446 · p < 0.0001 | 🔴 CRITICAL |
| Geographic disparity | DPD = 0.550 across state cohorts · p < 0.0001 | 🔴 CRITICAL |
| Gender disparity | DPD = 0.054 · p < 0.0001 | 🟡 MEDIUM |
| ICD9 code corruption robustness | F1 decay 7.7% at 20% corruption | 🟡 MEDIUM |
| Multi-field degradation | F1 decay 6.5% under combined pipeline failure | 🟡 MEDIUM |

**Overall governance verdict: RED — DO NOT DEPLOY without remediation.**

📄 [View full sample report →](reports/sample-report.html)

---

## What it evaluates

PayorLens runs five evaluation modules against any binary classification model on claims data:

**1. Data quality** — Pydantic v2 schema validation with field-level type coercion. Produces a data contract pass/fail before any model evaluation begins.

**2. Model performance** — Accuracy, F1, ROC-AUC, Brier score, confusion matrix. Flags high-confidence wrong predictions (>85% model confidence, wrong outcome) — the specific failure mode cited in the Cigna PxDx class-action litigation.

**3. Fairness audit** — Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD) across race, gender, age band, and geography. Every disparity metric is backed by a chi-square significance test. Nothing is flagged unless it clears both effect size and statistical significance thresholds.

**4. Robustness stress test** — Five clinically meaningful failure injection scenarios (not random noise): ICD9 code corruption, missing prior auth fields, age band enrollment lag, high-cost outlier claims, and combined multi-field degradation. Each scenario has a named clinical rationale from real payer data quality failure modes.

**5. Risk interpretation** — Every metric feeds through `risk_interpreter.py`, which maps findings to NIST AI RMF functions (Govern / Map / Measure / Manage) and subcategories, assigns a risk level (LOW / MEDIUM / HIGH / CRITICAL), writes a plain-English payor interpretation, and generates a recommended action. This is what makes the report readable to a compliance officer, not just an ML engineer.

---

## Report structure

The output is a two-audience HTML/PDF governance report:

| Section | Audience | Contents |
|---------|----------|----------|
| 0 · Executive Risk Brief | Compliance officer | Overall risk score, top 3 findings, single recommendation |
| 1 · NIST AI RMF Map | Legal / risk officer | Every metric → NIST function → PASS/WARN/FAIL |
| 2 · Data Quality | Data engineer | Pydantic validation results, error rate |
| 3 · Model Performance | ML engineer | F1, AUC, Brier, calibration diagram |
| 4–5 · Fairness Audit | Compliance + ML | Per-cohort DPD/EOD with p-values and risk narratives |
| 6 · Robustness | ML + compliance | F1 decay per clinical scenario, danger threshold |
| 7 · Failure Narratives | Compliance officer | Top 5 high-confidence errors as plain-language vignettes |
| 8 · Methodology | Auditor | Dataset provenance, statistical test rationale |

---

## Stack

```
Python 3.11+
scikit-learn        — model training and evaluation pipelines
fairlearn           — MetricFrame for per-cohort fairness metrics
scipy.stats         — chi2_contingency for all significance tests
pydantic v2         — schema validation and data contracts
pandas / numpy      — data wrangling
matplotlib          — charts (ROC curve, calibration, fairness bar charts)
joblib              — model serialization
typer               — CLI interface
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place CMS DE-SynPUF files in data/raw/cms/
#    Beneficiary Summary + Inpatient Claims (Sample 1)
#    Free registration: cms.gov/Research-Statistics-Data-and-Systems

# 3. Run the full pipeline
python cli.py evaluate \
  --bene-file DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv \
  --claims-file DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv \
  --model logistic

# Output: reports/payorlens_logistic.html
```

---

## Project structure

```
payorlens/
├── loader.py           # CMS data ingestion, normalization, Pydantic validation
├── evaluator.py        # Model training + core performance metrics
├── fairness.py         # FairnessAuditor — DPD/EOD with chi-square significance
├── robustness.py       # ClinicalRobustnessInjector — 5 failure scenarios
├── risk_interpreter.py # RiskInterpreter — metrics → risk narratives + NIST mapping
├── reporter.py         # ReportGenerator — two-audience HTML/PDF output
├── cli.py              # Typer CLI entry point
data/
├── raw/cms/            # CMS DE-SynPUF source files (not committed — too large)
├── processed/          # Parquet, trained models, charts
reports/
├── payorlens_logistic.html   # Sample report — logistic regression
├── payorlens_gbm.html        # Sample report — gradient boosting
```

---

## Regulatory alignment

All findings are mapped to the [NIST AI Risk Management Framework 1.0](https://airc.nist.gov/RMF) (January 2023), specifically:

- **MS-2.3** — AI output reliability and uncertainty quantification (calibration)
- **MS-2.5** — Bias and fairness testing across demographic cohorts
- **MS-2.6** — Robustness and resilience under input perturbation
- **MG-2.4** — High-consequence error routing and human oversight
- **MP-2.3** — Data provenance, quality, and lineage

NIST AI RMF is a voluntary federal framework. References to state statutes (CO SB21-169, NY Circular Letter No. 7) and NAIC guidance in report narratives are contextual illustrations of the regulatory environment a payer would face — not legal opinions.

---

## Methodology note

**On the target variable:** CMS DE-SynPUF does not include actual prior authorization denial decisions — that data is proprietary to individual payers. The `denial_status` label used in this evaluation is an engineered proxy, constructed using clinically informed logistic regression on utilization days, claim amount, chronic condition burden, and demographic features, calibrated to a ~21% denial rate consistent with published industry benchmarks.

The proxy label is used to demonstrate evaluation methodology. A production engagement would use a payer's actual denial labels. The framework — schema validation, fairness metrics, robustness scenarios, risk interpretation, NIST mapping — applies unchanged regardless of how the target variable is sourced.

**On model performance:** The logistic regression baseline (F1=0.379, AUC=0.632) is intentionally a starting point, not a production candidate. The purpose of this evaluation is to demonstrate what the governance framework catches when a model has problems — and this one has several. That is the point.

---

## Scope boundary

PayorLens is an **evaluation harness** — it assesses a model and produces a governance report. It is not a model registry, a compliance artifact generator, a circuit breaker, or a HITL workflow system. It evaluates one model at a time against public data and tells you what the risks are.

---

## Background

Built as an independent portfolio project demonstrating AI governance methodology for payer AI use cases. Domain informed by experience in health AI and regulatory compliance. Not affiliated with any payer, EHR vendor, or AI company.

---

*NIST AI RMF is a product of the National Institute of Standards and Technology. CMS DE-SynPUF is a public dataset from the Centers for Medicare & Medicaid Services.*
