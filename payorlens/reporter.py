# payorlens/reporter.py
"""
PayorLens ReportGenerator
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("PayorLens.Reporter")


RISK_COLORS = {
    "LOW"     : "#1E7A4A",
    "MEDIUM"  : "#C8932A",
    "HIGH"    : "#E67E22",
    "CRITICAL": "#B03A2E",
    "GREEN"   : "#1E7A4A",
    "AMBER"   : "#C8932A",
    "RED"     : "#B03A2E",
}

RISK_BG = {
    "LOW"     : "#E8F8EE",
    "MEDIUM"  : "#FDF6E3",
    "HIGH"    : "#FEF0E7",
    "CRITICAL": "#FDECEA",
    "GREEN"   : "#E8F8EE",
    "AMBER"   : "#FDF6E3",
    "RED"     : "#FDECEA",
}

RISK_EMOJI = {
    "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴",
    "GREEN": "🟢", "AMBER": "🟡", "RED": "🔴",
}

NIST_FUNCTIONS = {
    "Govern" : "#0D1F3C",
    "Map"    : "#0A7EA4",
    "Measure": "#C8932A",
    "Manage" : "#1E7A4A",
}


def _badge(level: str, text: Optional[str] = None) -> str:
    label = text or level
    color = RISK_COLORS.get(level, "#333")
    bg    = RISK_BG.get(level, "#f5f5f5")
    emoji = RISK_EMOJI.get(level, "⚪")
    return (f'<span style="background:{bg};color:{color};border:1px solid {color};'
            f'padding:3px 10px;border-radius:4px;font-weight:700;font-size:12px;">'
            f'{emoji} {label}</span>')


def _card(title: str, value: str, subtitle: str = "", color: str = "#0A7EA4") -> str:
    return f"""
    <div style="background:white;border-top:4px solid {color};border-radius:6px;
                padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,.08);">
        <div style="font-size:12px;color:#6B7280;text-transform:uppercase;
                    letter-spacing:.05em;margin-bottom:6px;">{title}</div>
        <div style="font-size:28px;font-weight:700;color:{color};">{value}</div>
        <div style="font-size:12px;color:#6B7280;margin-top:4px;">{subtitle}</div>
    </div>"""


def _section_header(title: str, color: str = "#0D1F3C") -> str:
    return f"""
    <h2 style="font-family:Arial,sans-serif;color:{color};font-size:18px;
               border-bottom:3px solid {color};padding-bottom:6px;margin-top:36px;">{title}</h2>"""


def _finding_block(finding_dict: dict) -> str:
    rl     = finding_dict.get("risk_level", "LOW")
    color  = RISK_COLORS.get(rl, "#333")
    bg     = RISK_BG.get(rl, "#f5f5f5")
    return f"""
    <div style="border-left:5px solid {color};background:{bg};padding:14px 18px;
                margin:12px 0;border-radius:0 6px 6px 0;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong style="font-size:14px;color:#0D1F3C;">
                {finding_dict.get('metric_name','Metric')}
            </strong>
            {_badge(rl)}
        </div>
        <div style="margin:8px 0;font-size:13px;color:#1F2937;">
            {finding_dict.get('payor_interpretation', '')}
        </div>
        <div style="font-size:12px;color:#374151;background:rgba(0,0,0,.04);
                    padding:6px 10px;border-radius:4px;margin-top:6px;">
            <strong>Recommended Action:</strong> {finding_dict.get('recommended_action','')}
        </div>
        <div style="font-size:11px;color:#6B7280;margin-top:6px;">
            NIST AI RMF: <strong>{finding_dict.get('nist_function','')}</strong>
            › {finding_dict.get('nist_subcategory','')} — {finding_dict.get('nist_description','')}
            &nbsp;|&nbsp; {finding_dict.get('statistical_note','')}
        </div>
    </div>"""


def _nist_table_row(finding: dict) -> str:
    fn_color = NIST_FUNCTIONS.get(finding.get("nist_function", ""), "#333")
    rl       = finding.get("risk_level", "LOW")
    return f"""
    <tr style="border-bottom:1px solid #DDE1E7;">
        <td style="padding:8px 10px;font-size:12px;">{finding.get('metric_name','')}</td>
        <td style="padding:8px 10px;">
            <span style="background:{NIST_BG(finding.get('nist_function',''))};
                   color:{fn_color};border:1px solid {fn_color};
                   padding:2px 8px;border-radius:3px;font-size:11px;font-weight:700;">
                {finding.get('nist_function','')}
            </span>
        </td>
        <td style="padding:8px 10px;font-size:11px;color:#374151;">
            {finding.get('nist_subcategory','')}
        </td>
        <td style="padding:8px 10px;">{_badge(rl)}</td>
    </tr>"""


def NIST_BG(fn: str) -> str:
    return {"Govern":"#E6EAF2","Map":"#E6F4F9","Measure":"#FDF6E3","Manage":"#E8F8EE"}.get(fn,"#F5F5F5")



class ReportGenerator:

    def __init__(self, output_dir: str = "D:/payorlens/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, report_data: dict, filename_stem: str = "payorlens_report") -> Path:
        html = self._build_html(report_data)
        html_path = self.output_dir / f"{filename_stem}.html"
        html_path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report → {html_path}")
        return html_path

    def render_pdf(self, report_data: dict, filename_stem: str = "payorlens_report") -> Optional[Path]:
        html_path = self.render(report_data, filename_stem)
        pdf_path  = self.output_dir / f"{filename_stem}.pdf"
        try:
            from weasyprint import HTML
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
            logger.info(f"PDF report → {pdf_path}")
            return pdf_path
        except ImportError:
            logger.warning("weasyprint not installed — PDF skipped. Run: pip install weasyprint")
            return None
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return None

    
    def _build_html(self, d: dict) -> str:
        exec_s   = d.get("executive_summary", {})
        findings = d.get("findings", [])
        eval_r   = d.get("eval_results", {})
        fair_r   = d.get("fairness_results", {})
        robust_r = d.get("robustness_results", {})
        dq_r     = d.get("data_quality", {})
        meta     = d.get("meta", {})
        narratives = d.get("failure_narratives", [])

        overall   = exec_s.get("overall_risk", "AMBER")
        ov_color  = RISK_COLORS.get(overall, "#333")
        ov_bg     = RISK_BG.get(overall, "#FDF6E3")
        ov_emoji  = RISK_EMOJI.get(overall, "⚠️")
        ts        = meta.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
        model_name= meta.get("model_name", "logistic")

        
        finding_html = "".join(_finding_block(f) for f in findings)

        
        nist_rows = "".join(_nist_table_row(f) for f in findings)

        
        fairness_html = self._build_fairness_section(fair_r, model_name)

        
        robust_html = self._build_robustness_section(robust_r, model_name)

        
        top_f = exec_s.get("top_findings", [])
        top_f_html = "".join(
            f'<div style="background:#F4F6F8;border-left:4px solid {ov_color};'
            f'padding:10px 14px;margin:8px 0;border-radius:0 4px 4px 0;font-size:13px;">'
            f'<strong>Finding {i+1}:</strong> {txt}</div>'
            for i, txt in enumerate(top_f)
        )

       
        best_model = eval_r.get(model_name, {})
        metric_cards = ""
        if best_model:
            metric_cards = f"""
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:16px 0;">
                {_card("Accuracy",    f"{best_model.get('accuracy',0):.3f}",    "Overall correctness",   "#0A7EA4")}
                {_card("F1 Score",    f"{best_model.get('f1',0):.3f}",          "Precision-Recall balance","#0D1F3C")}
                {_card("ROC-AUC",     f"{best_model.get('roc_auc',0):.3f}",     "Discrimination ability", "#1E7A4A")}
                {_card("Brier Score", f"{best_model.get('brier_score',0):.3f}", "Calibration quality",    "#C8932A")}
            </div>"""

        dq_total  = dq_r.get("total_records", 0)
        dq_errors = dq_r.get("error_count",   0)
        dq_rate   = dq_errors / max(dq_total, 1)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PayorLens AI Governance Report — {model_name}</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ font-family:Arial,sans-serif; color:#111827; margin:0; padding:0; background:#F4F6F8; }}
  .page {{ max-width:960px; margin:0 auto; padding:32px 24px; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ background:#0D1F3C; color:white; padding:10px; text-align:left; font-size:12px; }}
  td {{ padding:8px 10px; font-size:12px; vertical-align:top; }}
  tr:nth-child(even) {{ background:#F8F9FA; }}
  .section-box {{ background:white; border-radius:8px; padding:24px; margin:20px 0;
                  box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  @media print {{ body {{ background:white; }} .page {{ padding:16px; }} }}
</style>
</head>
<body>
<div class="page">

  <!-- ═══════════════ HEADER ═══════════════ -->
  <div style="background:#0D1F3C;color:white;padding:28px 32px;border-radius:8px 8px 0 0;">
    <div style="font-size:28px;font-weight:700;letter-spacing:-.5px;">PAYORLENS</div>
    <div style="font-size:14px;color:#A8D8EA;margin-top:4px;">
      AI Governance Evaluation Report · Model: {model_name} · {ts}
    </div>
    <div style="font-size:12px;color:#6B9DBA;margin-top:6px;">
      Dataset: CMS DE-SynPUF Inpatient Claims · Architecture v2.0 ·
      NIST AI RMF aligned
    </div>
  </div>

  <!-- ═══════════════ SECTION 0: EXECUTIVE RISK BRIEF ═══════════════ -->
  <div class="section-box" style="border-top:6px solid {ov_color};">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
      <div>
        <div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:.08em;">
          Section 0 · Executive Risk Brief
        </div>
        <h1 style="font-size:22px;margin:8px 0;color:#0D1F3C;">
          Overall Governance Risk Assessment
        </h1>
      </div>
      <div style="background:{ov_bg};border:2px solid {ov_color};padding:12px 24px;
                  border-radius:6px;text-align:center;">
        <div style="font-size:32px;">{ov_emoji}</div>
        <div style="font-size:18px;font-weight:700;color:{ov_color};">{overall}</div>
        <div style="font-size:11px;color:#6B7280;">Risk Level</div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0;">
      {_card("Total Findings",    str(exec_s.get("total_findings",0)),  "All risk levels",         "#0D1F3C")}
      {_card("Critical",          str(exec_s.get("critical_count",0)),  "Require immediate action","#B03A2E")}
      {_card("High",              str(exec_s.get("high_count",0)),      "Require remediation",     "#E67E22")}
      {_card("Medium / Low",
             f"{exec_s.get('medium_count',0)} / {exec_s.get('low_count',0)}",
             "Monitor",  "#1E7A4A")}
    </div>

    <div style="background:{ov_bg};border:1px solid {ov_color};padding:14px 18px;
                border-radius:6px;margin:16px 0;">
      <strong style="color:{ov_color};">Recommendation:</strong>
      <span style="font-size:13px;"> {exec_s.get("recommendation","")}</span>
    </div>

    <div style="margin-top:16px;">
      <strong style="font-size:13px;color:#0D1F3C;">Top Findings:</strong>
      {top_f_html}
    </div>
  </div>

  <!-- ═══════════════ SECTION 1: NIST AI RMF MAP ═══════════════ -->
  <div class="section-box">
    {_section_header("Section 1 · NIST AI RMF Compliance Mapping", "#0D1F3C")}
    <p style="font-size:13px;color:#374151;">
      Every evaluated metric is mapped to its corresponding NIST AI RMF function
      and subcategory. This table is the regulatory spine of the report.
    </p>
    <table>
      <thead><tr>
        <th>Metric</th><th>NIST Function</th><th>Subcategory</th><th>Status</th>
      </tr></thead>
      <tbody>{nist_rows}</tbody>
    </table>
  </div>

  <!-- ═══════════════ SECTION 2: DATA QUALITY ═══════════════ -->
  <div class="section-box">
    {_section_header("Section 2 · Data Quality Findings", "#0A7EA4")}
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:16px 0;">
      {_card("Total Records",   f"{dq_total:,}",              "After merge & normalisation", "#0A7EA4")}
      {_card("Validation Errors", f"{dq_errors:,}",           "Pydantic schema failures",    "#B03A2E")}
      {_card("Error Rate",      f"{dq_rate*100:.2f}%",        "Schema compliance rate",      "#C8932A")}
    </div>
    <p style="font-size:13px;color:#374151;">
      Data contract enforced via Pydantic v2 schema validation. Each record validated against
      ClaimRecord model with field-level type coercion and range checks.
      <strong>Note on CMS DE-SynPUF:</strong> SP_STATE_CODE and CLM_ID are integer-typed in
      raw CMS files — coerced to str in normalisation layer before schema validation.
    </p>
  </div>

  <!-- ═══════════════ SECTION 3: MODEL PERFORMANCE ═══════════════ -->
  <div class="section-box">
    {_section_header("Section 3 · Model Performance Overview", "#0A7EA4")}
    {metric_cards}
    <p style="font-size:12px;color:#6B7280;margin-top:12px;">
      High-confidence errors (>85% confidence, wrong prediction):
      <strong>{best_model.get('high_conf_wrong_count', 0)}</strong>
      ({best_model.get('high_conf_wrong_rate', 0)*100:.2f}% of test set).
      These are the 'dangerous prediction' events — model was highly confident AND wrong.
      In prior auth workflows, these become automated adverse determinations without human review.
    </p>
  </div>

  <!-- ═══════════════ SECTION 4–5: FAIRNESS + RISK FINDINGS ═══════════════ -->
  <div class="section-box">
    {_section_header("Section 4–5 · Fairness Audit & Risk Findings", "#C8932A")}
    {finding_html}
  </div>

  <!-- ═══════════════ FAIRNESS COHORT DETAIL ═══════════════ -->
  {fairness_html}

  <!-- ═══════════════ SECTION 6: ROBUSTNESS ═══════════════ -->
  <div class="section-box">
    {_section_header("Section 6 · Robustness Stress Test — Clinical Failure Scenarios", "#B03A2E")}
    {robust_html}
  </div>

  <!-- ═══════════════ SECTION 7: TOP FAILURE CASES ═══════════════ -->
  <div class="section-box">
    {_section_header("Section 7 · Top 5 High-Confidence Failure Cases", "#B03A2E")}
    <p style="font-size:12px;color:#6B7280;margin-bottom:12px;">
      The five predictions where the model was most confident AND wrong.
      In a prior auth automation workflow these become unreviewed adverse determinations.
      Each narrative includes patient profile, error type, confidence level,
      and governance implication.
    </p>
    {"".join(
        f'<div style="background:#FEF0E7;border-left:5px solid #B03A2E;padding:14px 18px;'
        f'margin:10px 0;border-radius:0 6px 6px 0;font-size:13px;line-height:1.75;">{n}</div>'
        for n in narratives
    ) if narratives else
    '<p style="color:#6B7280;font-size:13px;font-style:italic;">No failure narratives generated — re-run pipeline after leakage fix is applied.</p>'}
  </div>

  <!-- ═══════════════ SECTION 8: METHODOLOGY ═══════════════ -->
  <div class="section-box">
    {_section_header("Section 8 · Methodology & Robustness Threshold Legend", "#0D1F3C")}
    <ul style="font-size:13px;color:#374151;line-height:1.8;">
      <li><strong>Dataset:</strong> CMS DE-SynPUF Inpatient Claims + Beneficiary Summary,
          joined on DESYNPUF_ID. Engineered denial target: logit model over utilization,
          chronic conditions, age, demographic weights + noise (σ=1.2) → ~20% base rate.</li>
      <li><strong>Target rationale:</strong> CLM_PMT_AMT==0 proxy removed — it caused
          target leakage (AUC=1.0). Engineered target reflects realistic prior auth
          denial patterns per CMS/NAIC literature.</li>
      <li><strong>Model:</strong> {model_name} with class_weight="balanced".
          claim_amount excluded from features (was leaking into target).</li>
      <li><strong>Fairness tests:</strong> scipy chi-square on each sensitive feature.
          Findings only flagged if p &lt; 0.05 AND |DPD| ≥ 0.05. HIGH minimum if |DPD| ≥ 0.10.</li>
      <li><strong>Robustness thresholds:</strong>
          🟢 Safe &lt;5% F1 decay &nbsp;|&nbsp;
          🟡 Warning 5–10% &nbsp;|&nbsp;
          🟠 High 10–20% &nbsp;|&nbsp;
          🔴 Danger &gt;20% F1 decay.</li>
      <li><strong>NIST AI RMF:</strong> Voluntary framework (not Radinate-proprietary).
          Mappings based on NIST AI RMF v1.0 published guidance.</li>
    </ul>
  </div>

  <!-- ═══════════════ FOOTER ═══════════════ -->
  <div style="background:#0D1F3C;color:#A8D8EA;padding:16px 24px;border-radius:0 0 8px 8px;
              font-size:11px;text-align:center;margin-top:8px;">
    PayorLens AI Governance Evaluation Harness · Architecture v2.0 · Generated {ts} ·
    NIST AI RMF aligned · Dataset: CMS DE-SynPUF (public, zero PHI) ·
    This report is an independent evaluation artefact. It is not a state compliance document.
  </div>

</div>
</body>
</html>"""
        return html

    
    def _build_fairness_section(self, fair_r: dict, model_name: str) -> str:
        if not fair_r:
            return ""
        model_data = fair_r.get(model_name, fair_r)
        html = ""
        for feature, res in model_data.items():
            dpd   = res.get("dpd", 0)
            sig   = "✅ Significant" if res.get("statistically_significant") else "⚪ Not significant"
            rl    = res.get("risk_level", "LOW")
            color = RISK_COLORS.get(rl, "#333")
            bg    = RISK_BG.get(rl, "#f5f5f5")

            by_group = res.get("by_group", {})
            rows = ""
            for group, vals in by_group.items():
                dr  = vals.get("denial_rate")
                cnt = vals.get("count", 0)
                f1  = vals.get("f1")
                note= vals.get("note", "")
                rows += f"""<tr>
                  <td>{group}</td>
                  <td>{cnt:,}</td>
                  <td>{"N/A" if dr is None else f"{dr:.3f}"}</td>
                  <td>{"N/A" if f1 is None else f"{f1:.3f}"}</td>
                  <td>{note or "—"}</td>
                </tr>"""

            html += f"""
            <div class="section-box" style="border-left:5px solid {color};">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <h3 style="font-size:15px;color:#0D1F3C;margin:0;">
                  Fairness — {feature.replace("_"," ").title()}
                </h3>
                {_badge(rl)}
              </div>
              <div style="font-size:12px;color:#374151;margin:8px 0;">
                DPD={dpd:.4f} &nbsp;|&nbsp; EOD={res.get('eod',0):.4f} &nbsp;|&nbsp;
                χ² p={res.get('chi2_pvalue',1):.4f} &nbsp;|&nbsp; {sig}
              </div>
              <table>
                <thead><tr>
                  <th>Cohort</th><th>Count</th><th>Denial Rate</th>
                  <th>F1</th><th>Note</th>
                </tr></thead>
                <tbody>{rows}</tbody>
              </table>
            </div>"""
        return html

    def _build_robustness_section(self, robust_r: dict, model_name: str) -> str:
        if not robust_r:
            return "<p>No robustness results available.</p>"
        data = robust_r.get(model_name, robust_r)
        baseline = data.get("baseline_f1", 0)
        scenarios= data.get("scenarios", {})

        rows = ""
        for key, res in scenarios.items():
            decay    = res.get("decay_pct", 0)
            danger   = res.get("danger_threshold_exceeded", False)
            warning  = res.get("warning_threshold_exceeded", False)
            rl       = "CRITICAL" if decay > 25 else "HIGH" if decay > 20 else "MEDIUM" if decay > 10 else "LOW" if decay > 5 else "LOW"
            threshold_flag = "🔴 DANGER" if danger else "🟡 WARNING" if warning else "✅ OK"
            rows += f"""<tr>
              <td>{res.get('description','')}</td>
              <td>{res.get('injection_rate',0)*100:.0f}%</td>
              <td>{res.get('baseline_f1',0):.3f}</td>
              <td>{res.get('degraded_f1',0):.3f}</td>
              <td><strong>{decay:.1f}%</strong></td>
              <td style="white-space:nowrap">{threshold_flag}</td>
              <td>{_badge(rl)}</td>
            </tr>"""

        return f"""
        <p style="font-size:12px;color:#6B7280;">
          Baseline F1: <strong>{baseline:.4f}</strong> &nbsp;·&nbsp;
          🟢 Safe &lt;5% &nbsp;|&nbsp; 🟡 Warning 5–10% &nbsp;|&nbsp;
          🟠 High 10–20% &nbsp;|&nbsp; 🔴 Danger &gt;20% F1 decay
        </p>
        <table>
          <thead><tr>
            <th>Scenario</th><th>Rate</th><th>Baseline F1</th>
            <th>Degraded F1</th><th>Decay %</th><th>Threshold</th><th>Risk</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>"""



def assemble_report_data(
    eval_results:      dict,
    fairness_results:  dict,
    robustness_results:dict,
    all_findings:      list,
    executive_summary,
    data_quality:      dict,
    model_name:        str,
    failure_narratives: list = None,
) -> dict:
    from risk_interpreter import RiskFinding
    findings_dicts = [f.__dict__ if hasattr(f, "__dict__") else f for f in all_findings]
    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model_name"  : model_name,
            "framework"   : "NIST AI RMF v1.0",
            "dataset"     : "CMS DE-SynPUF Inpatient Claims",
        },
        "executive_summary"  : executive_summary.__dict__ if hasattr(executive_summary, "__dict__") else executive_summary,
        "findings"           : findings_dicts,
        "eval_results"       : eval_results,
        "fairness_results"   : fairness_results,
        "robustness_results" : robustness_results,
        "data_quality"       : data_quality,
        "failure_narratives" : failure_narratives or [],
    }