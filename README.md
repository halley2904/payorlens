PayorLens

Enterprise AI Governance & Audit Harness for Clinical Claims Models

🔗 Interactive Dashboard: payorlens.streamlit.app

🔗 API Documentation (Swagger): payorlens.onrender.com/docs

📄 Sample Executive Governance Report: halley2904.github.io/payorlens

Note on Live Demos: Both the API backend (Render) and frontend (Streamlit Cloud) run on free-tier infrastructure. If idle, the backend service spins down automatically. Initial requests may take 30–60 seconds to wake up from a cold start.

1. Product Overview & Strategic Context

Who Is This For?

Chief Risk & Compliance Officers: Needing clear audit trails, legal defensibility, and NIST-aligned risk verdicts before approving AI model deployments in claims workflows.

Clinical AI Product Managers: Responsible for ensuring model behavior aligns with clinical policy, operational reliability, and patient equity targets.

ML Engineers & Data Scientists: Looking to move beyond offline F1 scores and evaluate models against real-world clinical failure modes, cohort disparities, and schema drift.

The Problem (Why)

Health plans are increasingly automating prior authorization and claims adjudication using AI models. However, a major disconnect exists between model development and executive risk approval:

Metrics don't communicate operational or legal risk. A data scientist sees a Demographic Parity Difference (DPD) of 0.286 or a p-value $< 0.0001$ and recognizes a statistical anomaly. A compliance officer or legal counsel needs to know: Does this violate CMS fair practice rules? Will this trigger regulatory action or litigation?

Regulatory & Litigation Pressure. High-profile class-action lawsuits (e.g., Cigna PxDx litigation) and regulatory frameworks (NAIC Algorithmic Information Request, CMS-4201-F) prioritize organizational governance: Did the health plan detect, document, and remediate systemic bias and automated denial patterns before shipping?

The Solution & Impact (What)

PayorLens is an end-to-end evaluation and governance harness that sits between clinical model outputs and production deployment. It ingests model inference data, executes automated fairness and clinical stress tests, translates statistical outputs into plain-English risk narratives, and generates a two-audience governance report.

De-risks AI Deployment: Automatically flags high-confidence incorrect denial decisions (a primary driver of bad-faith denial lawsuits).

Bridges Tech & Legal: Maps raw statistical metrics directly to NIST AI RMF functions (Govern, Map, Measure, Manage) with automated risk determinations.

Enforces Hard Data Contracts: Prevents pipeline corruption by validating input schemas using strict Pydantic rules prior to evaluation.

2. System Architecture

                      ┌────────────────────────────────────────┐
                      │    Streamlit Dashboard (Frontend)      │
                      │       payorlens.streamlit.app          │
                      └──────────────────┬─────────────────────┘
                                         │ REST API
                                         ▼
                      ┌────────────────────────────────────────┐
                      │      FastAPI Service (Render)          │ ──► SQLite (Run History &
                      │       payorlens.onrender.com           │      Structured Metrics)
                      └──────────────────┬─────────────────────┘
                                         │ In-Process Pipeline Call
                                         ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ Evaluation Core                                                                        │
 │  [Data Loader] ──► [Model Evaluator] ──► [Fairness Audit] ──► [Clinical Stress Test]   │
 │         │                                                             │                │
 │         └─────────────────────────────┬───────────────────────────────┘                │
 │                                       ▼                                                │
 │                           [NIST Risk Interpreter]                                      │
 └───────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
                                         ▼
                             [Guardrailed LLM Narrator]
                      (Translates metrics into plain-English)
                                         │
                                         ▼
                         [HTML / PDF Audit Report Output]


Component

Responsibility

Technical Stack

Evaluation Core

Schema validation, performance auditing, fairness quantification, stress testing, NIST mapping, and HTML report rendering

scikit-learn, Fairlearn, scipy, Pydantic v2, Jinja2

API Backend

Wraps the core pipeline as REST endpoints (POST /evaluate, GET /runs, GET /runs/{id}); manages run persistence

FastAPI, Uvicorn, SQLite

LLM Narrator

Converts structured evaluation metrics into plain-English risk summaries for non-technical stakeholders using strict numerical guardrails

Anthropic Claude API / Google Gemini API

User Dashboard

Interactive interface allowing non-technical users to trigger evaluation runs, inspect cohort metrics, and view generated reports

Streamlit, Plotly

3. What PayorLens Evaluates

Data Contract Integrity

Field-level Pydantic validation and type coercion. Rejects corrupt or incomplete claims payloads before running downstream metrics.

Clinical Performance & High-Confidence Failures

Tracks standard metrics (F1, ROC-AUC, Brier score). Crucially flags high-confidence wrong predictions ($>85\%$ confidence score on incorrect denials), isolating the specific failure pattern behind automated claims lawsuits.

Protected Cohort Fairness Audit

Evaluates Demographic Parity Difference (DPD) and Equalized Odds Difference (EOD) across protected attributes (race, gender, age band, geography). Every flagged disparity requires both an effect-size threshold breach and a statistically significant chi-square test ($p < 0.05$).

Clinical Failure Injections (Robustness)

Subject models to 5 real-world edge cases rather than random gaussian noise:

ICD-9 code corruption

Missing prior-authorization details

Age-band enrollment lag

High-cost outlier claims

Multi-field combined degradation

NIST AI RMF Risk Mapping

Automatically categorizes findings into NIST functions (Govern, Map, Measure, Manage), determines overall risk levels (Low, Medium, High, Critical), and generates action items for executive review.

4. Sample Evaluation Findings

Based on CMS DE-SynPUF inpatient claims (66,718 records) evaluated on a Logistic Regression baseline.

Audit Finding

Underlying Metric

Risk Level

Governance Verdict

Model Calibration Failure

Brier Score = 0.236 · 8 high-confidence wrong decisions

🔴 CRITICAL

Model overconfident on bad predictions

Race Cohort Disparity

DPD = 0.286 ($p < 0.0001$)

🔴 CRITICAL

Unacceptable bias in denial distribution

Age Band Disparity

DPD = 0.446 ($p < 0.0001$)

🔴 CRITICAL

Older age bands disproportionately denied

Geographic Disparity

DPD = 0.550 across state cohorts ($p < 0.0001$)

🔴 CRITICAL

Regional variance in model decisions

Gender Disparity

DPD = 0.054 ($p < 0.0001$)

🟡 MEDIUM

Moderate disparity requiring monitoring

ICD-9 Corruption Robustness

F1 decay of 7.7% under 20% data corruption

🟡 MEDIUM

Model degraded by input field formatting

Multi-Field Degradation

F1 decay of 6.5% under combined failure injection

🟡 MEDIUM

System performance degrades under partial data

Overall Executive Verdict: 🔴 RED — DO NOT DEPLOY WITHOUT REMEDIATION

5. Key Architecture & PM Tradeoffs

Strategic Decision

Choice Made

PM Rationale

Tradeoff / Limitation

Pipeline Execution

In-process execution inside the API handler

Prevents opaque subprocess exit codes; provides full tracebacks and direct metric access for real-time risk assessment

Lack of process isolation (a crashing evaluation can impact API availability)

Data Persistence

SQLite file database

Zero operational overhead for single-instance demo; rapid prototyping

Limited concurrent write capability

Request Model

Synchronous POST /evaluate endpoint

Dataset is fixed and evaluation takes seconds; avoids unneeded queue infrastructure for single-instance workloads

Cannot gracefully handle concurrent long-running requests without a background queue (e.g., Celery)

Deployment Setup

Native Python host (Render & Streamlit Cloud)

Direct deployment eliminates Docker overhead for demo scale

Requires manual environment alignment across cloud hosts

LLM Guardrails

Deterministic strict numeric verification

Prevents hallucinated metrics in compliance documents; rejects summaries containing numbers not present in raw metrics

Narratives can fallback to template text if the LLM output violates guardrails

Public API Scope

Fixed CMS sample dataset

Ensures fast, deterministic evaluation runs and prevents arbitrary file upload exploits on free-tier servers

Users cannot upload arbitrary custom datasets via the live public web demo

6. Report Structure

PayorLens renders a unified HTML/PDF report structured specifically to balance compliance overview with technical depth:

Section

Target Audience

Key Contents

0. Executive Risk Brief

Chief Compliance Officer, Legal

Overall risk color (Red/Yellow/Green), top critical findings, deployment approval recommendation

1. NIST AI RMF Mapping

Risk Officer, Auditor

Finding-to-NIST mapping (Govern, Map, Measure, Manage) with pass/fail indicators

2. Data Quality & Schema

Data Engineer, ML Engineer

Pydantic validation status, missing field ratios, schema contract health

3. Model Performance

ML Engineer

ROC-AUC, F1-score, Brier calibration score, high-confidence error counts

4–5. Fairness & Equity Audit

Compliance Officer, ML Engineer

Cohort-level DPD/EOD metrics, chi-square significance values, plain-language risk commentary

6. Clinical Robustness

ML Engineer, Clinical PM

Performance decay curves across 5 failure scenario injections

7. Failure Case Vignettes

Compliance Officer, Product Manager

Concrete plain-language vignettes of top high-confidence wrong decisions

8. Methodology & Provenance

External Auditor

Data sources, statistical assumptions, reproducible evaluation configuration

7. Future Roadmap

Automated Drift & Version Comparison (GET /runs/compare): Diff two separate evaluation runs to highlight metric regressions or fairness drift over time.

Background Task Queue & Scalable Storage: Integrate Celery/Redis with PostgreSQL to handle multi-tenant concurrent evaluations.

Custom Ingestion Pipelines: Support dynamic S3/GCS file uploads with user-configurable schema mappers for private payer datasets.

Role-Based Access Control (RBAC): Multi-tenant auth ensuring clinical teams, auditors, and engineering groups maintain appropriate data visibility.

8. Technical Stack & Local Setup

Core Stack

Python: 3.11+

ML & Statistics: scikit-learn, fairlearn, scipy, pandas, numpy

Validation & API: pydantic v2, fastapi, uvicorn

Visualization & Frontend: streamlit, plotly, matplotlib

LLM Layer: anthropic / google-generativeai

Quick Start (Local Development)

Bash

# 1. Clone repo and install dependencies
git clone https://github.com/halley2904/payorlens.git
cd payorlens
pip install -r requirements.txt

# 2. Place CMS DE-SynPUF dataset files in data/raw/cms/
#    (Beneficiary Summary + Inpatient Claims files)

# 3a. Option A: Run via CLI directly
python cli.py evaluate \
  --bene-file DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv \
  --claims-file DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv \
  --model logistic

# 3b. Option B: Run the local API backend
uvicorn payorlens_api:app --reload
# Submit evaluation request:
curl -X POST http://localhost:8000/evaluate -H "Content-Type: application/json" -d '{"model":"logistic"}'

# 3c. Option C: Run the interactive dashboard (requires running API)
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py


Directory Layout

payorlens/
├── loader.py              # Ingestion, schema normalization, Pydantic validation
├── evaluator.py           # Model training & standard metric calculations
├── fairness.py            # Cohort fairness auditor (DPD/EOD + Chi-square tests)
├── robustness.py          # Clinical robustness stress-testing scenarios
├── risk_interpreter.py    # Risk interpretation & NIST AI RMF mapping engine
├── reporter.py            # HTML/PDF dual-audience report generator
├── narrator.py            # Guardrailed LLM plain-English summary layer
├── payorlens_api.py       # FastAPI REST service & persistence layer
├── streamlit_app.py       # Streamlit interactive dashboard UI
├── cli.py                 # Command-line interface entry point
├── data/                  # Raw and processed dataset files
└── reports/               # Generated HTML governance reports


PayorLens was developed as an independent portfolio project demonstrating applied AI PM thinking, software engineering, and AI governance methodology for healthcare payer workflows. It is not affiliated with any specific insurance payer, EHR vendor, or cloud provider.