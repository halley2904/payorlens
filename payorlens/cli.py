"""
PayorLens CLI
"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    print("ERROR: typer not installed. Run: pip install typer")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("PayorLens.CLI")

app = typer.Typer(
    name="payorlens",
    help="PayorLens — AI Governance Evaluation Harness for Payor/Insurance AI",
    add_completion=False,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RAW_DIR       = "D:/payorlens/data/raw/cms"
DEFAULT_PROCESSED_DIR = "D:/payorlens/data/processed"
DEFAULT_REPORTS_DIR   = "D:/payorlens/reports"


# ── validate command ──────────────────────────────────────────────────────────
@app.command()
def validate(
    bene_file  : str = typer.Option(..., help="Beneficiary CSV filename (in raw/cms/)"),
    claims_file: str = typer.Option(..., help="Claims CSV filename (in raw/cms/)"),
    raw_dir    : str = typer.Option(DEFAULT_RAW_DIR,       help="Raw data directory"),
    out_dir    : str = typer.Option(DEFAULT_PROCESSED_DIR, help="Processed output directory"),
):
    """
    Day 1 & 2: Load and validate CMS DE-SynPUF data.
    Runs the loader, normalises, validates with Pydantic, saves parquet.
    """
    from loader import CMSLoader
    loader = CMSLoader(raw_dir=raw_dir, processed_dir=out_dir)
    df = loader.process(bene_file, claims_file)
    typer.echo(f"\n Validation complete.")
    typer.echo(f"    Valid records : {len(df):,}")
    typer.echo(f"    Errors        : {len(loader.validation_errors):,}")
    typer.echo(f"    Parquet saved : {out_dir}/claims_v1.parquet")


# ── evaluate command ──────────────────────────────────────────────────────────
@app.command()
def evaluate(
    bene_file  : Optional[str] = typer.Option(None,  help="Beneficiary CSV filename"),
    claims_file: Optional[str] = typer.Option(None,  help="Claims CSV filename"),
    parquet    : Optional[str] = typer.Option(None,  help="Processed parquet path (skip loading)"),
    raw_dir    : str = typer.Option(DEFAULT_RAW_DIR,       help="Raw data directory"),
    proc_dir   : str = typer.Option(DEFAULT_PROCESSED_DIR, help="Processed data directory"),
    reports_dir: str = typer.Option(DEFAULT_REPORTS_DIR,   help="Reports output directory"),
    model      : str = typer.Option("logistic",            help="Model: logistic | gbm"),
    pdf        : bool= typer.Option(False,                 help="Also generate PDF"),
    samples    : int = typer.Option(0,                     help="Sample N rows (0=use all)"),
):
    """
    Full PayorLens evaluation pipeline:
    Load → Eval → Fairness → Robustness → Risk Interpret → Report
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    parquet_path = parquet or f"{proc_dir}/claims_v1.parquet"

    if not Path(parquet_path).exists():
        if not bene_file or not claims_file:
            typer.echo("ERROR: No parquet found and no CSV files provided.", err=True)
            raise typer.Exit(1)
        from loader import CMSLoader
        typer.echo("  Loading and processing CMS data …")
        loader = CMSLoader(raw_dir=raw_dir, processed_dir=proc_dir)
        df = loader.process(bene_file, claims_file)
        dq = {
            "total_records": len(df) + len(loader.validation_errors),
            "valid_records": len(df),
            "error_count"  : len(loader.validation_errors),
        }
    else:
        typer.echo(f" Loading parquet: {parquet_path}")
        df  = pd.read_parquet(parquet_path)
        dq  = {"total_records": len(df), "valid_records": len(df), "error_count": 0}
        loader = None

    if samples and samples < len(df):
        df = df.sample(samples, random_state=42)
        typer.echo(f" Sampled {samples:,} rows for evaluation")

    # ── Step 2: Train & Evaluate ──────────────────────────────────────────────
    # claim_amount removed — was leaking into denial_status target (AUC=1.0 bug)
    NUMERIC_FEATURES     = ["age","utilization_days","deductible_amount","chronic_count",
                             "has_diabetes","has_chf","has_copd","has_cancer"]
    CATEGORICAL_FEATURES = ["gender","race","age_band","state_code"]
    TARGET = "denial_status"

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    # keep full df rows aligned with X_test for narrative extraction
    df_test = df.loc[X_test.index].copy()

    typer.echo(f"\n Training {model} model …")
    from evaluator import EvalEngine
    engine = EvalEngine(output_dir=proc_dir)
    eval_results = engine.train_and_evaluate(parquet_path)
    trained_model = engine.models[model]

    # ── Step 3: Fairness audit ────────────────────────────────────────────────
    typer.echo(f"\n Running fairness audit …")
    from fairness import FairnessAuditor
    y_pred = trained_model.predict(X_test)
    auditor = FairnessAuditor(output_dir=proc_dir)
    fairness_results_raw = auditor.audit(y_test, y_pred, X_test, model)
    auditor.generate_all_cohort_charts(model)

    # ── Step 4: Robustness ────────────────────────────────────────────────────
    typer.echo(f"\n  Running robustness stress tests …")
    from robustness import RobustnessEvaluator
    rob_evaluator  = RobustnessEvaluator(output_dir=proc_dir)
    robustness_raw = rob_evaluator.run_all(trained_model, X_test, y_test, model)

    # ── Step 4b: Top-5 failure narratives ────────────────────────────────────
    typer.echo(f"\n Extracting top-5 failure narratives …")
    em_raw = eval_results.get(model, {})
    y_true_list = em_raw.pop("_y_true", list(y_test))
    y_pred_list = em_raw.pop("_y_pred", list(trained_model.predict(X_test)))
    y_prob_list = em_raw.pop("_y_prob", list(trained_model.predict_proba(X_test)[:,1]))

    import numpy as _np
    y_true_arr = _np.array(y_true_list)
    y_pred_arr = _np.array(y_pred_list)
    y_prob_arr = _np.array(y_prob_list)

    # High-confidence wrong: model confident AND wrong
    wrong_mask = y_pred_arr != y_true_arr
    confidence = _np.where(y_pred_arr == 1, y_prob_arr, 1 - y_prob_arr)
    wrong_conf  = confidence * wrong_mask
    top5_idx    = _np.argsort(wrong_conf)[::-1][:5]
    test_indices = X_test.index.tolist()

    failure_narratives = []
    CONDITION_MAP = {
        "has_diabetes":"Diabetes","has_chf":"Congestive Heart Failure",
        "has_copd":"COPD","has_cancer":"Cancer"
    }
    for rank, i in enumerate(top5_idx, 1):
        orig_idx = test_indices[i]
        row = df_test.loc[orig_idx]
        true_label = int(y_true_arr[i])
        pred_label = int(y_pred_arr[i])
        conf       = float(confidence[i])
        conditions = [label for col,label in CONDITION_MAP.items()
                      if row.get(col,0)==1]
        cond_str = ", ".join(conditions) if conditions else "no flagged conditions"
        outcome_true = "DENIED" if true_label == 1 else "APPROVED"
        outcome_pred = "DENIED" if pred_label == 1 else "APPROVED"
        error_type   = "False Denial" if pred_label==1 and true_label==0 else "False Approval"
        narrative = (
            f"<strong>Rank {rank} — {error_type}</strong> "
            f"(model confidence: {conf*100:.1f}%)<br>"
            f"Patient: {row.get('gender','Unknown')}, age {int(row.get('age',0))}, "
            f"race: {row.get('race','Unknown')}, state: {row.get('state_code','?')}<br>"
            f"Chronic conditions: {cond_str} | "
            f"Utilization: {int(row.get('utilization_days',0))} days<br>"
            f"Model predicted: <strong>{outcome_pred}</strong> · "
            f"Actual outcome: <strong>{outcome_true}</strong><br>"
            f"<em>Governance implication: "
            + (
                f"A {error_type.lower()} at {conf*100:.1f}% confidence bypasses human review "
                f"in an automated prior auth workflow, becoming an unreviewed adverse determination. "
                f"This patient profile ({row.get('race','Unknown')}, age {int(row.get('age',0))}) "
                f"represents a demographic group with elevated model error rates per fairness audit."
                if error_type == "False Denial" else
                f"A {error_type.lower()} at {conf*100:.1f}% confidence allows a potentially "
                f"non-covered service through without review, creating financial exposure for the payer."
            ) + "</em>"
        )
        failure_narratives.append(narrative)

    # ── Step 5: Risk interpretation ───────────────────────────────────────────
    typer.echo(f"\n Interpreting risk findings …")
    from risk_interpreter import RiskInterpreter
    interpreter = RiskInterpreter()
    all_findings = []

    # Data quality
    er = dq["error_count"] / max(dq["total_records"], 1)
    all_findings.append(interpreter.interpret_data_quality(er, dq["total_records"]))

    # Calibration / high-confidence errors
    em = eval_results.get(model, {})
    all_findings.append(interpreter.interpret_calibration(
        em.get("brier_score", 0),
        em.get("high_conf_wrong_count", 0),
        em.get("total_test_samples", 1),
    ))

    # Fairness per feature
    for feature, res in fairness_results_raw.items():
        all_findings.append(interpreter.interpret_dpd(
            res["dpd"], feature, res["chi2_pvalue"], model
        ))

    # Robustness per scenario
    baseline_f1 = robustness_raw.get("baseline_f1", 0)
    for scen_key, scen_res in robustness_raw.get("scenarios", {}).items():
        all_findings.append(interpreter.interpret_robustness(
            scen_res["description"],
            baseline_f1,
            scen_res["degraded_f1"],
            scen_res["injection_rate"],
        ))

    exec_summary = interpreter.generate_executive_summary(all_findings, model)

    # ── Step 6: Report ────────────────────────────────────────────────────────
    typer.echo(f"\n Generating report …")
    from reporter import ReportGenerator, assemble_report_data

    report_data = assemble_report_data(
        eval_results       = eval_results,
        fairness_results   = {model: fairness_results_raw},
        robustness_results = {model: robustness_raw},
        all_findings       = all_findings,
        executive_summary  = exec_summary,
        data_quality       = dq,
        model_name         = model,
        failure_narratives = failure_narratives,
    )

    gen = ReportGenerator(output_dir=reports_dir)
    html_path = gen.render(report_data, filename_stem=f"payorlens_{model}")
    typer.echo(f"HTML report → {html_path}")

    if pdf:
        pdf_path = gen.render_pdf(report_data, filename_stem=f"payorlens_{model}")
        if pdf_path:
            typer.echo(f" PDF report  → {pdf_path}")
        else:
            typer.echo(" PDF skipped ")

    # ── Summary ───────────────────────────────────────────────────────────────
    typer.echo("\n" + "="*60)
    typer.echo("  PAYORLENS EVALUATION COMPLETE")
    typer.echo("="*60)
    typer.echo(f"  Model          : {model}")
    typer.echo(f"  Overall Risk   : {exec_summary.overall_risk}")
    typer.echo(f"  Critical       : {exec_summary.critical_count}")
    typer.echo(f"  High           : {exec_summary.high_count}")
    typer.echo(f"  Recommendation : {exec_summary.recommendation[:80]}…")
    typer.echo("="*60)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app()