# PayorLens

**AI governance evaluation harness for clinical decision models in insurance workflows**

🔗 **Live dashboard**: _add your Streamlit Cloud URL here_
🔗 **API docs (Swagger)**: _add your AWS URL + `/docs` here_
📄 **Sample report**: [halley2904.github.io/payorlens](https://halley2904.github.io/payorlens/)

---

## 1. Problem Statement

Health insurers are deploying AI in prior authorization and claims adjudication and decisions that directly affect whether a patient gets covered care. There is a gap consistently show up in practice:

1. **Metrics don't translate into consequence.** A technical person sees `DPD = 0.286, p < 0.0001` and knows it's bad. A compliance officer, legal counsel, or a product manager approving a launch does not  and they're the ones who actually decide whether the model ships. The Cigna PxDx litigation and NAIC's Algorithmic Information Request framework both center on exactly this: not just "was the model biased," but "did the organization know, document, and act on it."

**PayorLens addresses it.** The core evaluation harness (`loader.py` → `evaluator.py` → `fairness.py` → `robustness.py` → `risk_interpreter.py` → `reporter.py`) produces a two-audience governance report readable by a compliance officer, actionable . On top of that, a FastAPI service and Streamlit dashboard make it a callable, persisted, non-technical-user-triggerable tool.

## 2. System Overview

```
                    ┌─────────────────────┐
   Dashboard  ────► │   FastAPI service    │ ────► SQLite (run history,
 (Streamlit Cloud)   │  (payorlens_api.py)  │        structured metrics)
                    └──────────┬───────────┘
                               │ in-process calls
                               ▼
              loader → evaluator → fairness → robustness
                     → risk_interpreter → reporter
                               │
                               ▼
                    LLM narrator (narrator.py)
              guardrailed plain-English risk summary
```

| Component | Responsibility | Tech |
|---|---|---|
| Evaluation core | Data validation, model training, fairness audit, robustness stress test, NIST-mapped risk interpretation, HTML/PDF report generation | scikit-learn, Fairlearn, scipy, Pydantic v2, Jinja2 |
| API service | Wraps the evaluation core as `POST /evaluate`, `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/report`; persists run history | FastAPI, SQLite |
| LLM narrator | Turns structured metrics into a plain-English summary for non-technical stakeholders, with a hard guardrail against fabricated numbers | Anthropic/Gemini API |
| Dashboard | Interactive, non-technical trigger + visualization layer over the API | Streamlit, Plotly |

## 3. What It Evaluates

1. **Data quality** — Pydantic v2 schema validation with field-level type coercion. Pass/fail data contract before any model evaluation begins.
2. **Model performance** — Accuracy, F1, ROC-AUC, Brier score, confusion matrix. Flags high-confidence wrong predictions (>85% confidence, wrong outcome) — the specific failure mode cited in the Cigna PxDx litigation.
3. **Fairness audit** — Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD) across race, gender, age band, geography. Every disparity is backed by a chi-square significance test — nothing is flagged unless it clears both effect-size and significance thresholds.
4. **Robustness stress test** — Five clinically meaningful failure injections (not random noise): ICD9 code corruption, missing prior-auth fields, age-band enrollment lag, high-cost outlier claims, combined multi-field degradation.
5. **Risk interpretation** — Maps every finding to NIST AI RMF functions (Govern/Map/Measure/Manage), assigns a risk level, writes a plain-English payer interpretation, and generates a recommended action.

## 4. Sample Findings

_Evaluated on CMS DE-SynPUF inpatient claims (66,718 records) · Logistic Regression baseline_

| Finding | Metric | Risk Level |
|---|---|---|
| Model calibration failure | Brier = 0.236 · 8 high-confidence wrong predictions | 🔴 CRITICAL |
| Race cohort denial disparity | DPD = 0.286 · p < 0.0001 | 🔴 CRITICAL |
| Age band denial disparity | DPD = 0.446 · p < 0.0001 | 🔴 CRITICAL |
| Geographic disparity | DPD = 0.550 across states · p < 0.0001 | 🔴 CRITICAL |
| Gender disparity | DPD = 0.054 · p < 0.0001 | 🟡 MEDIUM |
| ICD9 code corruption robustness | F1 decay 7.7% at 20% corruption | 🟡 MEDIUM |
| Multi-field degradation | F1 decay 6.5% under combined failure | 🟡 MEDIUM |

**Overall governance verdict: RED — DO NOT DEPLOY without remediation.**

## 5. Design Decisions & Tradeoffs


| Decision | What I chose | Why | What I gave up |
|---|---|---|---|
| Pipeline execution model | Run the evaluation pipeline **in-process** inside the API request handler, not as a subprocess calling `cli.py` | A subprocess-based version hid real errors behind opaque exit codes and was fragile across OS/path differences. In-process gives full tracebacks and direct access to structured metrics objects | Process isolation - because a crashing evaluation could theoretically affect the API process.|
| Persistence | SQLite, single file, no ORM | Zero operational overhead for a single-instance, low-write-volume service. Migrating to Postgres later is a one-line change, nothing to rewrite | Concurrent-writer safety. First thing I'd change if this needed to serve multiple simultaneous evaluators |
| Request handling | Synchronous `POST /evaluate` (blocks until the run finishes) | The dataset is small and fixed; a run takes seconds to low minutes. Adding a job queue for a workload this size would be infrastructure nobody's using yet | Won't handle concurrent long-running requests gracefully so  a background task queue (Celery/RQ) the moment concurrent usage is real in future |
| Deployment | Direct Python deploy, not Docker | Docker added real debugging overhead for zero benefit at single-instance scale. Docker artifacts are kept in the repo for future portability, just not on the deploy path today | Docker would provide environment parity |
| LLM narrative layer | The narrator can only use numbers present in the input metrics and any output containing a number that isn't in the source metrics is discarded and replaced with a deterministic fallback | A governance tool cannot afford a hallucinated compliance statistic. Trustworthiness of the number matters more than narrative richness | Some runs get a plainer, template-based summary instead of a fuller LLM narrative. Correct tradeoff for this domain |
| Public API dataset | Fixed CMS sample dataset, no arbitrary file upload endpoint | Keeps the public demo API fast, stateless-ish, and safe from unbounded upload abuse on a free-tier server | Not usable against a client's actual data as-is.|
| Dashboard framework | Streamlit, not a custom frontend | Standard tool for exposing a Python/AI backend without frontend engineering scope | Less visual polish/control than a custom frontend.|

## 6. What I'd Change With More Resources

1. **Model/run comparison endpoint** (`GET /runs/compare?a=&b=`): version of drift detection: diff two runs' metrics, flag regressions
2. **Postgres + a background job queue** —  to handle more than one evaluator at a time.
3. **Real data ingestion** — Replacing the fixed sample dataset, so this could run against an actual client's claims data.
4. **CI/CD** - GitHub Actions auto-deploy on push to main
6. **Multi-tenant auth** —  if this was for evaluating multiple payers' models under one deployment

## 7. Report Structure

| Section | Audience | Contents |
|---|---|---|
| 0 · Executive Risk Brief | Compliance officer | Overall risk score, top 3 findings, single recommendation |
| 1 · NIST AI RMF Map | Legal / risk officer | Every metric → NIST function → PASS/WARN/FAIL |
| 2 · Data Quality | Data engineer | Pydantic validation results, error rate |
| 3 · Model Performance | ML engineer | F1, AUC, Brier, calibration diagram |
| 4–5 · Fairness Audit | Compliance + ML | Per-cohort DPD/EOD with p-values and risk narratives |
| 6 · Robustness | ML + compliance | F1 decay per clinical scenario, danger threshold |
| 7 · Failure Narratives | Compliance officer | Top 5 high-confidence errors as plain-language vignettes |
| 8 · Methodology | Auditor | Dataset provenance, statistical test rationale |

## 8. Stack

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
fastapi / uvicorn   — API service layer
streamlit / plotly  — interactive dashboard
anthropic / gemini  — LLM narrator with guardrailed output
```

## 9. Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Placing CMS DE-SynPUF files in data/raw/cms/
#    Beneficiary Summary + Inpatient Claims (Sample 1)
#    can get it for free by registring: cms.gov/Research-Statistics-Data-and-Systems

# 3a. Run via CLI directly
python cli.py evaluate \
  --bene-file DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv \
  --claims-file DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv \
  --model logistic

# 3b. OR run via the API
uvicorn payorlens_api:app --reload
curl -X POST localhost:8000/evaluate -H "Content-Type: application/json" -d '{"model":"logistic"}'

# 3c. OR run the dashboard (needs the API running)
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

## 10. Project Structure

```
payorlens/
├── loader.py                  # CMS data ingestion, normalization, Pydantic validation
├── evaluator.py                # Model training + core performance metrics
├── fairness.py                 # FairnessAuditor - DPD/EOD with chi-square significance
├── robustness.py                # ClinicalRobustnessInjector - 5 failure scenarios
├── risk_interpreter.py         # RiskInterpreter - metrics -> risk narratives + NIST mapping
├── reporter.py                  # ReportGenerator - two-audience HTML/PDF output
├── cli.py                       # Typer CLI entry point
├── payorlens_api.py             # FastAPI service wrapping the pipeline
├── narrator.py                  # Guardrailed LLM narrative layer
├── streamlit_app.py             # Interactive dashboard client
├── requirements.txt              # Pipeline + API dependencies
├── requirements-streamlit.txt    # Dashboard-only dependencies
data/
├── raw/cms/                     # CMS DE-SynPUF source files 
├── processed/                   # Parquet, trained models, charts
reports/
├── payorlens_logistic.html      # Sample report — logistic regression
├── payorlens_gbm.html           # Sample report — gradient boosting
```



Built as an independent portfolio project demonstrating AI governance methodology for payer AI use cases, extended into a deployable API + dashboard to demonstrate applied AI engineering and solutions/product thinking, not just modeling. Not affiliated with any payer, EHR vendor, or AI company.

NIST AI RMF is a product of the National Institute of Standards and Technology. CMS DE-SynPUF is a public dataset from the Centers for Medicare & Medicaid Services.
