"""
streamlit_app.py — PayorLens AI Governance Dashboard

"""

from __future__ import annotations

import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="PayorLens | AI Governance Audit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLORS = {
    "CRITICAL": "#E5484D",
    "HIGH": "#F5A524",
    "MEDIUM": "#F5D90A",
    "LOW": "#3DD68C",
    "UNKNOWN": "#6B7280",
}

CUSTOM_CSS = """
<style>
    /* Global Styles */
    .stApp {
        background-color: #0B0E14;
        color: #E6E6E6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Sidebar Polish */
    section[data-testid="stSidebar"] {
        background-color: #12161F;
        border-right: 1px solid #1E2430;
    }
    
    /* Executive Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #161B26 0%, #11151F 100%);
        border: 1px solid #232A3B;
        border-radius: 12px;
        padding: 20px;
        text-align: left;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #8C96A8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Business Narrative Callout */
    .narrative-card {
        background: rgba(229, 72, 77, 0.08);
        border: 1px solid rgba(229, 72, 77, 0.3);
        border-left: 5px solid #E5484D;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 16px 0 24px 0;
    }
    .narrative-card h4 {
        margin-top: 0;
        color: #FF6B6B;
        font-size: 1.1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .narrative-card p {
        color: #D1D5DB;
        margin-bottom: 8px;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Tab Customization */
    button[data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 500;
        color: #8C96A8;
        padding: 10px 16px;
    }
    button[aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom-color: #38BDF8 !important;
    }

    /* Custom Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-critical { background: rgba(229, 72, 77, 0.2); color: #FF6B6B; border: 1px solid #E5484D; }
    .badge-success { background: rgba(61, 214, 140, 0.2); color: #3DD68C; border: 1px solid #3DD68C; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#C9D1D9", family="sans-serif"),
    margin=dict(l=10, r=10, t=30, b=10),
)


def api_get(base_url: str, path: str, timeout: int = 15):
    resp = requests.get(f"{base_url.rstrip('/')}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def api_post(base_url: str, path: str, json_body: dict, timeout: int = 600):
    resp = requests.post(f"{base_url.rstrip('/')}{path}", json=json_body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def metric_card(label: str, value: str, color: str | None = None) -> str:
    color_style = f"color: {color};" if color else "color: #FFFFFF;"
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value" style="{color_style}">{value}</div>
    </div>
    """

st.sidebar.title("🛡️ PayorLens")
st.sidebar.caption("AI Governance & Audit Suite")

with st.sidebar.expander("⚙️ API Configuration", expanded=False):

    try:
        secret_api_url = st.secrets.get(
        "PAYORLENS_API_URL",
        "https://payorlens.onrender.com"
    )
    except Exception:
        secret_api_url = "https://payorlens.onrender.com"

    default_api_url = st.session_state.get("api_base_url", secret_api_url)
    api_base_url = st.text_input(
        "API Base URL",
        value=default_api_url,
        help="FastAPI Backend Endpoint",
    )
st.session_state["api_base_url"] = api_base_url

st.sidebar.markdown("---")
st.sidebar.subheader("New Evaluation Audit")

model_choice = st.sidebar.selectbox("Target Model", options=["logistic", "gbm"], index=0)

run_clicked = st.sidebar.button("▶ Run Audit Evaluation", use_container_width=True, type="primary")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Sync Audit State", use_container_width=True):
    st.session_state.pop("runs_cache", None)
    st.rerun()

if run_clicked:
    with st.spinner(f"Evaluating {model_choice} model against business governance rules…"):
        try:
            # Fixed sample parameter at 0 in backend call as requested
            result = api_post(api_base_url, "/evaluate", {"model": model_choice, "samples": 0})
            st.session_state["selected_run_id"] = result["id"]
            st.session_state.pop("runs_cache", None)
            st.sidebar.success(f"Audit Complete: {result['risk_verdict']}")
        except requests.exceptions.RequestException as exc:
            detail = getattr(exc.response, "text", str(exc)) if getattr(exc, "response", None) else str(exc)
            st.sidebar.error(f"Evaluation failed:\n\n{detail}")


st.title("PayorLens Model Governance Audit")
st.caption("Active risk oversight and compliance validation")

try:
    if "runs_cache" not in st.session_state:
        st.session_state["runs_cache"] = api_get(api_base_url, "/runs")
    runs = st.session_state["runs_cache"]
except requests.exceptions.RequestException as exc:
    st.error(
        f"Unable to connect to the PayorLens API (`{api_base_url}`). "
        f"Please check your service status."
    )
    st.stop()

if not runs:
    st.info("No audit runs recorded yet. Use **Run Audit Evaluation** in the sidebar to start.")
    st.stop()

succeeded_runs = [r for r in runs if r["status"] == "succeeded"]

run_options = {f"Audit Run — Model: {r['model_type']} ({r['status'].upper()})": r["id"] for r in runs}
default_id = st.session_state.get("selected_run_id") or (succeeded_runs[0]["id"] if succeeded_runs else runs[0]["id"])
default_label = next((label for label, rid in run_options.items() if rid == default_id), list(run_options.keys())[0])

selected_label = st.selectbox("Select Active Audit Evaluation", options=list(run_options.keys()), index=list(run_options.keys()).index(default_label))
selected_id = run_options[selected_label]

run = api_get(api_base_url, f"/runs/{selected_id}")

if run["status"] == "failed":
    st.error(f"Selected audit failed:\n\n{run.get('error_message') or 'No error detail available.'}")
    st.stop()

if run["status"] != "succeeded":
    st.warning("Audit evaluation is currently processing...")
    st.stop()

metrics = run["metrics"] or {}

overall_risk = (metrics.get("overall_risk") or "UNKNOWN").upper()
risk_color = RISK_COLORS.get(overall_risk, RISK_COLORS["UNKNOWN"])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(metric_card("Overall Risk Verdict", overall_risk, risk_color), unsafe_allow_html=True)
with col2:
    st.markdown(metric_card("Critical Findings", str(metrics.get("critical_count", "—")), RISK_COLORS["CRITICAL"]), unsafe_allow_html=True)
with col3:
    st.markdown(metric_card("High Risk Findings", str(metrics.get("high_count", "—")), RISK_COLORS["HIGH"]), unsafe_allow_html=True)
with col4:
    st.markdown(metric_card("Target Model", run["model_type"].upper(), "#38BDF8"), unsafe_allow_html=True)

narrative = metrics.get("ai_narrative")
if narrative:
    
    verdict_copy = {
        "CRITICAL": ("⛔", "DEPLOYMENT VERDICT: DO NOT DEPLOY"),
        "HIGH": ("⚠️", "DEPLOYMENT VERDICT: REMEDIATION REQUIRED"),
        "MEDIUM": ("🟡", "DEPLOYMENT VERDICT: REVIEW BEFORE DEPLOYING"),
        "LOW": ("✅", "DEPLOYMENT VERDICT: NO BLOCKING ISSUES FOUND"),
    }
    icon, verdict_title = verdict_copy.get(overall_risk, ("❔", "DEPLOYMENT VERDICT: UNKNOWN"))

    st.markdown(
        f"""
        <div class="narrative-card" style="border-left-color: {risk_color}; background: {risk_color}14; border-color: {risk_color}4D;">
            <h4 style="color: {risk_color};">{icon} {verdict_title}</h4>
            <p><strong>Executive Summary:</strong> {narrative.get('summary', '')}</p>
            <p><strong>Primary Vulnerability:</strong> {narrative.get('top_risk', '')}</p>
            <p><strong>Required Action:</strong> {narrative.get('recommended_action', '')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


tab_overview, tab_fairness, tab_performance, tab_history = st.tabs([
    "📊 Risk & Governance Overview", 
    "⚖️ Fairness & Bias Audit", 
    "📈 Operational Performance", 
    "📜 Audit History"
])

with tab_overview:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Demographic Bias Risk")
        fairness = metrics.get("fairness", {})
        if fairness:
            features = list(fairness.keys())
            dpds = [fairness[f]["dpd"] for f in features]
            colors = [RISK_COLORS.get(fairness[f]["risk_level"], RISK_COLORS["UNKNOWN"]) for f in features]

            fig = go.Figure(go.Bar(x=dpds, y=features, orientation="h", marker_color=colors))
            fig.update_layout(**PLOTLY_DARK_LAYOUT, xaxis_title="Demographic Parity Difference", height=280)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No fairness audit metrics recorded.")

    with chart_col2:
        st.subheader("Model Degradation Risks")
        robustness = metrics.get("robustness", {})
        scenarios = robustness.get("scenarios", {}) if robustness else {}
        if scenarios:
            names = list(scenarios.keys())
            decays = [scenarios[s]["decay_pct"] for s in names]
            colors = ["#E5484D" if scenarios[s]["danger_threshold_exceeded"] else "#3DD68C" for s in names]

            fig = go.Figure(go.Bar(x=decays, y=names, orientation="h", marker_color=colors))
            fig.update_layout(**PLOTLY_DARK_LAYOUT, xaxis_title="Performance Degradation Rate (%)", height=280)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No robustness audit metrics recorded.")

with tab_fairness:
    st.subheader("Demographic Parity Breakdown")
    st.markdown("Detailed audit regarding protected class variances and decision equity.")
    if fairness:
        for feat, data in fairness.items():
            st.write(f"**Feature:** `{feat}` — Parity Difference: `{data['dpd']:.3f}` | Status: **{data['risk_level']}**")

with tab_performance:
    perf_col, dq_col = st.columns(2)
    with perf_col:
        st.subheader("Model Business Metrics")
        perf = metrics.get("performance", {})
        if perf:
            
            st.markdown(f"- **Accuracy Level:** `{perf.get('accuracy', 'N/A')}` *(Requires > 0.85 for production)*")
            st.markdown(f"- **F1 Score Efficiency:** `{perf.get('f1', 'N/A')}` *(High error variance detected)*")
            st.markdown(f"- **AUC-ROC Stability:** `{perf.get('roc_auc', 'N/A')}`")

    with dq_col:
        st.subheader("Data Integrity Health")
        dq = metrics.get("data_quality", {})
        if dq:
            st.markdown(f"- **Total Audited Records:** `{dq.get('total_records', 'N/A')}`")
            st.markdown(f"- **Valid Decision Records:** `{dq.get('valid_records', 'N/A')}`")
            st.markdown(f"- **Critical System Errors:** `{dq.get('error_count', 0)}`")

with tab_history:
    st.subheader("Recent System Audits")
    st.markdown("Historical overview of model governance runs.")
    
    
    for idx, r in enumerate(runs):
        with st.container():
            h_col1, h_col2, h_col3 = st.columns([2, 2, 2])
            h_col1.markdown(f"**Model:** `{r['model_type'].upper()}`")
            
            status_style = "badge-success" if r['status'] == "succeeded" else "badge-critical"
            h_col2.markdown(f"<span class='status-badge {status_style}'>{r['status'].upper()}</span>", unsafe_allow_html=True)
            
            verdict = r.get('risk_verdict') or 'UNKNOWN'
            verdict_color = RISK_COLORS.get(verdict.upper(), "#9CA3AF")
            h_col3.markdown(f"Verdict: <strong style='color:{verdict_color}'>{verdict}</strong>", unsafe_allow_html=True)
            st.divider()


st.write("")
report_url = f"{api_base_url.rstrip('/')}/runs/{selected_id}/report"
st.link_button("📄 Open Full Governance Compliance Report", report_url, use_container_width=True)