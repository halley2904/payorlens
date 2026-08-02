"""
PayorLens API — FastAPI service that runs the evaluation pipeline in-process.

"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from narrator import generate_narrative

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("payorlens.api")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PIPELINE_DIR = Path(os.getenv("PAYORLENS_PIPELINE_DIR", str(PROJECT_ROOT))).resolve()

RAW_DIR = Path(os.getenv("PAYORLENS_RAW_DIR", str(PIPELINE_DIR / "data" / "raw"))).resolve()


PROC_DIR = Path(os.getenv("PAYORLENS_PROC_DIR", str(PIPELINE_DIR / "data" / "processed"))).resolve()


REPORTS_DIR = Path(os.getenv("PAYORLENS_REPORTS_DIR", str(PIPELINE_DIR / "reports"))).resolve()


DB_PATH = Path(os.getenv("PAYORLENS_DB_PATH", str(PIPELINE_DIR / "payorlens_runs.db"))).resolve()


ALLOWED_MODELS = ("logistic", "gbm")
DEFAULT_BENE_FILE = "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv"
DEFAULT_CLAIMS_FILE = "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

logger.info("PIPELINE_DIR = %s", PIPELINE_DIR)
logger.info("RAW_DIR      = %s", RAW_DIR)
logger.info("PROC_DIR     = %s", PROC_DIR)
logger.info("REPORTS_DIR  = %s", REPORTS_DIR)


RUN_STATUS_RUNNING = "running"
RUN_STATUS_SUCCEEDED = "succeeded"
RUN_STATUS_FAILED = "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    model_type    TEXT NOT NULL,
    dataset_name  TEXT NOT NULL,
    status        TEXT NOT NULL,
    risk_verdict  TEXT,
    metrics_json  TEXT,
    report_path   TEXT,
    error_message TEXT
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.execute(_SCHEMA)


def insert_run(run_id: str, model_type: str, dataset_name: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO runs (id, created_at, model_type, dataset_name, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, datetime.now(timezone.utc).isoformat(), model_type, dataset_name, RUN_STATUS_RUNNING),
        )


def mark_run_succeeded(run_id: str, risk_verdict: str | None, metrics: dict, report_path: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, risk_verdict = ?, metrics_json = ?, report_path = ? WHERE id = ?",
            (RUN_STATUS_SUCCEEDED, risk_verdict, json.dumps(metrics, default=str), report_path, run_id),
        )


def mark_run_failed(run_id: str, error_message: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, error_message = ? WHERE id = ?",
            (RUN_STATUS_FAILED, error_message, run_id),
        )


def fetch_run(run_id: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()


def fetch_all_runs() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()


def row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["metrics"] = json.loads(d.pop("metrics_json")) if d.get("metrics_json") else None
    return d



class PipelineError(RuntimeError):
    """Wraps any failure during pipeline execution with a clear message,
    so the API can return something you can actually debug from."""


def _require_file(path: Path, what: str) -> None:
    if not path.exists():
        raise PipelineError(
            f"{what} not found at expected path: {path}\n"
            f"(Checked because PAYORLENS_RAW_DIR / PAYORLENS_PROC_DIR resolved to this location — "
            f"set the env var explicitly if your data lives elsewhere.)"
        )


def run_pipeline(model: str, bene_file: str | None, claims_file: str | None, samples: int = 0) -> tuple[str, dict]:
    """Runs load -> train/eval -> fairness -> robustness -> risk interpret -> report,
    exactly as cli.py's evaluate() does, and returns (report_path, metrics_dict)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    NUMERIC_FEATURES = ["age", "utilization_days", "deductible_amount", "chronic_count",
                        "has_diabetes", "has_chf", "has_copd", "has_cancer"]
    CATEGORICAL_FEATURES = ["gender", "race", "age_band", "state_code"]
    TARGET = "denial_status"

    proc_dir = Path(PROC_DIR)
    parquet_path = proc_dir / "claims_v1.parquet"

    
    if not parquet_path.exists():
        if not bene_file or not claims_file:
            raise PipelineError(
                f"No cached parquet at {parquet_path} and no bene_file/claims_file provided. "
                f"First call needs both — supply them in the request body."
            )
        raw_path = Path(RAW_DIR)
        _require_file(raw_path / bene_file, "bene_file")
        _require_file(raw_path / claims_file, "claims_file")

        from loader import CMSLoader
        logger.info("Loading + validating CMS data (bene=%s, claims=%s) …", bene_file, claims_file)
        loader = CMSLoader(raw_dir=str(raw_path), processed_dir=str(proc_dir))
        df = loader.process(bene_file, claims_file)
        dq = {
            "total_records": len(df) + len(loader.validation_errors),
            "valid_records": len(df),
            "error_count": len(loader.validation_errors),
        }
    else:
        logger.info("Using cached parquet: %s", parquet_path)
        df = pd.read_parquet(parquet_path)
        dq = {"total_records": len(df), "valid_records": len(df), "error_count": 0}

    if samples and samples < len(df):
        df = df.sample(samples, random_state=42)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    
    logger.info("Training %s model …", model)
    from evaluator import EvalEngine
    engine = EvalEngine(output_dir=str(proc_dir))
    eval_results = engine.train_and_evaluate(str(parquet_path))
    trained_model = engine.models[model]

    
    logger.info("Running fairness audit …")
    from fairness import FairnessAuditor
    y_pred = trained_model.predict(X_test)
    auditor = FairnessAuditor(output_dir=str(proc_dir))
    fairness_results_raw = auditor.audit(y_test, y_pred, X_test, model)
    auditor.generate_all_cohort_charts(model)

    
    logger.info("Running robustness stress tests …")
    from robustness import RobustnessEvaluator
    rob_evaluator = RobustnessEvaluator(output_dir=str(proc_dir))
    robustness_raw = rob_evaluator.run_all(trained_model, X_test, y_test, model)

    
    logger.info("Interpreting risk findings …")
    from risk_interpreter import RiskInterpreter
    interpreter = RiskInterpreter()
    all_findings = []

    error_rate = dq["error_count"] / max(dq["total_records"], 1)
    all_findings.append(interpreter.interpret_data_quality(error_rate, dq["total_records"]))

    em = dict(eval_results.get(model, {}))
    all_findings.append(interpreter.interpret_calibration(
        em.get("brier_score", 0),
        em.get("high_conf_wrong_count", 0),
        em.get("total_test_samples", 1),
    ))

    for feature, res in fairness_results_raw.items():
        all_findings.append(interpreter.interpret_dpd(res["dpd"], feature, res["chi2_pvalue"], model))

    baseline_f1 = robustness_raw.get("baseline_f1", 0)
    for scen_key, scen_res in robustness_raw.get("scenarios", {}).items():
        all_findings.append(interpreter.interpret_robustness(
            scen_res["description"], baseline_f1, scen_res["degraded_f1"], scen_res["injection_rate"],
        ))

    exec_summary = interpreter.generate_executive_summary(all_findings, model)

    
    logger.info("Generating report …")
    from reporter import ReportGenerator, assemble_report_data
    report_data = assemble_report_data(
        eval_results=eval_results,
        fairness_results={model: fairness_results_raw},
        robustness_results={model: robustness_raw},
        all_findings=all_findings,
        executive_summary=exec_summary,
        data_quality=dq,
        model_name=model,
        failure_narratives=[],  
    )
    gen = ReportGenerator(output_dir=REPORTS_DIR)
    html_path = gen.render(report_data, filename_stem=f"payorlens_{model}")

    
    perf = {k: v for k, v in em.items() if not k.startswith("_")}
    metrics = {
        "data_quality": dq,
        "performance": perf,
        "fairness": {
            feature: {
                "dpd": res["dpd"],
                "eod": res["eod"],
                "chi2_pvalue": res["chi2_pvalue"],
                "risk_level": res["risk_level"],
            }
            for feature, res in fairness_results_raw.items()
        },
        "robustness": {
            "baseline_f1": robustness_raw.get("baseline_f1"),
            "scenarios": {
                key: {
                    "decay_pct": s["decay_pct"],
                    "danger_threshold_exceeded": s["danger_threshold_exceeded"],
                }
                for key, s in robustness_raw.get("scenarios", {}).items()
            },
        },
        "overall_risk": getattr(exec_summary, "overall_risk", None),
        "critical_count": getattr(exec_summary, "critical_count", None),
        "high_count": getattr(exec_summary, "high_count", None),
        "recommendation": getattr(exec_summary, "recommendation", None),
    }

    return str(html_path), metrics



class EvaluateRequest(BaseModel):
    model: Literal["logistic", "gbm"] = "logistic"
    bene_file: Optional[str] = Field(default=None, description="Only needed before a parquet is cached")
    claims_file: Optional[str] = Field(default=None, description="Only needed before a parquet is cached")
    samples: int = Field(default=0, description="0 = use all rows")


class RunResponse(BaseModel):
    id: str
    created_at: str
    model_type: str
    dataset_name: str
    status: str
    risk_verdict: str | None = None
    metrics: dict | None = None
    report_path: str | None = None
    error_message: str | None = None



app = FastAPI(
    title="PayorLens API",
    version="0.2.0",
    description="Runs the PayorLens evaluation pipeline in-process and persists results.",
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(PROC_DIR).mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/evaluate", response_model=RunResponse, status_code=201)
def evaluate(payload: EvaluateRequest) -> RunResponse:
    if payload.model not in ALLOWED_MODELS:
        raise HTTPException(400, f"Unsupported model '{payload.model}'. Allowed: {ALLOWED_MODELS}")

    bene_file = payload.bene_file or DEFAULT_BENE_FILE
    claims_file = payload.claims_file or DEFAULT_CLAIMS_FILE
    dataset_name = f"{bene_file} + {claims_file}"

    run_id = str(uuid.uuid4())
    insert_run(run_id, payload.model, dataset_name)

    try:
        report_path, metrics = run_pipeline(payload.model, bene_file, claims_file, payload.samples)
    except PipelineError as exc:
        msg = str(exc)
        mark_run_failed(run_id, msg)
        logger.error("Pipeline failed for run %s: %s", run_id, msg)
        raise HTTPException(502, detail=msg) from exc
    except Exception as exc:  # noqa: BLE001 — deliberately broad: surface EVERYTHING while debugging
        tb = traceback.format_exc()
        logger.error("Unhandled exception during run %s:\n%s", run_id, tb)
        mark_run_failed(run_id, tb)
        raise HTTPException(500, detail=f"{type(exc).__name__}: {exc}\n\n{tb[-3000:]}") from exc

    
    metrics["ai_narrative"] = generate_narrative(metrics)

    mark_run_succeeded(run_id, metrics.get("overall_risk"), metrics, report_path)
    return RunResponse(**row_to_dict(fetch_run(run_id)))


@app.get("/runs", response_model=list[RunResponse])
def list_runs() -> list[RunResponse]:
    return [RunResponse(**row_to_dict(row)) for row in fetch_all_runs()]


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> RunResponse:
    row = fetch_run(run_id)
    if row is None:
        raise HTTPException(404, detail="Run not found")
    return RunResponse(**row_to_dict(row))


@app.get("/runs/{run_id}/report")
def get_run_report(run_id: str):
    row = fetch_run(run_id)
    if row is None or not row["report_path"]:
        raise HTTPException(404, detail="Report not found for this run")
    path = Path(row["report_path"])
    if not path.exists():
        raise HTTPException(404, detail="Report file no longer exists on disk")
    return FileResponse(path, media_type="text/html")
