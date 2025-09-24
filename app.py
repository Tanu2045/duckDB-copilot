# ==== Environment & HTTP ====
import os
import requests
from dotenv import load_dotenv

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Defaults that work even if .env has nothing extra
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))  # seconds

def _ollama_ok() -> bool:
    try:
        r = requests.get(OLLAMA_BASE_URL.rstrip("/") + "/api/tags", timeout=(5, 5))
        return r.ok
    except Exception:
        return False

def _ollama_chat(payload: dict) -> str:
    """
    Centralized POST with reliable timeouts and tight generation settings
    (no env changes required).
    """
    url = OLLAMA_BASE_URL.rstrip("/") + "/api/chat"

    # Fast, deterministic defaults (can be overridden inside payload if needed)
    merged = {
        **payload,
        "model": payload.get("model", OLLAMA_MODEL),
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.1,
            "num_predict": 140,   # keep outputs short (bullets/short SQL)
            "num_ctx": 768,       # smaller context for speed
            **payload.get("options", {}),
        },
    }
    r = requests.post(url, json=merged, timeout=(10, OLLAMA_TIMEOUT))  # 10s connect, long read
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()

# ==== UI & Data ====
import streamlit as st
import pandas as pd
import duckdb

# ==== Local modules ====
from dq_checks import check_uniqueness, check_null_threshold, check_value_range
from prompts import SQL_SYSTEM, INSIGHTS_SYSTEM, EXEC_SUMMARY_SYSTEM  # INSIGHTS_SYSTEM kept for reference

# ==== Page config ====
st.set_page_config(page_title="DuckDB Copilot (WIP)", layout="wide")
st.title("DuckDB Copilot — NL→SQL, Data Quality, Report")
st.caption("Upload a CSV, ask questions in English → safe SQL in DuckDB → quality checks → executive summary → export report.")

def _auto_pick_insight_columns(df: pd.DataFrame):
    """
    Data-only picker:
      label_col: non-numeric, moderate-cardinality column for grouping
      measure_col: numeric column suitable for aggregation; None => use row count
    """
    import numpy as np
    n = len(df)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_num_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    # --- label_col: prefer 5..min(50, 0.3*n) distinct values (stable Top-N)
    upper_cap = max(5, int(0.3 * n)) if n else 50
    label_candidates = []
    for c in non_num_cols:
        k = int(df[c].nunique(dropna=True))
        if 5 <= k <= min(50, upper_cap):
            target = 15
            score = -abs(k - target)  # closer to ~15 is better
            label_candidates.append((score, c, k))
    if label_candidates:
        label_candidates.sort(key=lambda x: (x[0], x[1]))
        label_col = label_candidates[0][1]
    elif non_num_cols:
        label_col = non_num_cols[0]
    else:
        label_col = df.columns[0]

    # --- helpers to exclude bad metric candidates
    def is_id_like(v: pd.Series) -> bool:
        if n == 0:
            return False
        distinct = v.nunique(dropna=True)
        return (distinct >= 0.9 * n) or v.is_unique

    def is_timestamp_like(vals: pd.Series) -> bool:
        try:
            v = pd.to_numeric(vals, errors="coerce").dropna()
            if v.empty: return False
            lo, hi = float(v.min()), float(v.max())
            seconds_epoch = (8.5e8 <= lo <= 2.2e9) and (8.5e8 <= hi <= 2.2e9)
            millis_epoch  = (8.5e11 <= lo <= 2.2e12) and (8.5e11 <= hi <= 2.2e12)
            year_like     = (1900 <= lo <= 2100) and (1900 <= hi <= 2100)
            return seconds_epoch or millis_epoch or year_like
        except Exception:
            return False

    # --- measure_col: numeric, not ID/timestamp/binary/constant; has variance
    metric_cands = []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if v.empty: continue
        if is_id_like(v) or is_timestamp_like(v): continue
        nunique = v.nunique()
        if nunique <= 5: continue
        if float(v.max()) == float(v.min()): continue

        nonzero_frac = (v != 0).mean()
        nonneg_frac  = (v >= 0).mean()
        variance_ok  = 1.0 if (len(v) > 1 and float(np.var(v)) > 0.0) else 0.0
        distinct_ratio = nunique / max(1, len(v))
        id_penalty = 1.0 if distinct_ratio >= 0.9 else 0.0

        score = 2.0*nonzero_frac + 1.5*nonneg_frac + 1.0*(1.0-id_penalty) + 0.5*variance_ok
        metric_cands.append((score, c))

    if metric_cands:
        metric_cands.sort(key=lambda x: (-x[0], x[1]))
        measure_col = metric_cands[0][1]
    else:
        measure_col = None  # fall back to row count

    return label_col, measure_col

# ==== Helper: SQL validation ====
def validate_sql(sql_text: str) -> str:
    if not sql_text:
        raise ValueError("Empty SQL")
    s = sql_text.strip()
    if ";" in s:
        raise ValueError("Multiple statements are not allowed")
    low = s.lower()
    if not low.startswith("select"):
        raise ValueError("Only SELECT statements are allowed")
    forbidden = (" drop ", " insert ", " update ", " delete ", " pragma ", " attach ")
    padded = " " + low + " "
    if any(tok in padded for tok in forbidden):
        raise ValueError("Dangerous SQL blocked")
    return s

def _summarize_df_for_llm(df: pd.DataFrame, max_rows: int = 30) -> str:
    # schema
    schema = ", ".join([f"{c} ({str(df[c].dtype)})" for c in df.columns])
    rows, cols = df.shape

    # top nulls
    nulls_top = df.isna().sum().sort_values(ascending=False).head(10)
    null_text = "; ".join([f"{c}:{int(n)}" for c, n in nulls_top.items() if n > 0]) or "none"

    # numeric quick stats (compressed)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    num_stats_lines = []
    for c in num_cols[:10]:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if not s.empty:
            p95 = float(s.quantile(0.95))
            num_stats_lines.append(
                f"{c}: mean={s.mean():.2f}, p50={s.median():.2f}, p95={p95:.2f}, min={s.min():.2f}, max={s.max():.2f}"
            )
    num_stats = "\n".join(num_stats_lines) or "none"

    # top categories (non-numeric) – quick feel for distributions
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    cat_samples = []
    for c in cat_cols[:6]:
        vc = df[c].value_counts(dropna=True).head(5)
        if not vc.empty:
            cat_samples.append(f"{c}: " + ", ".join([f"{str(k)}({int(v)})" for k, v in vc.items()]))
    cats_text = "\n".join(cat_samples) or "none"

    # tiny sample rows to ground the model
    sample_csv = df.head(max_rows).to_csv(index=False)

    return (
        f"Schema: {schema}\n"
        f"Rows: {rows}, Cols: {cols}\n"
        f"Top nulls: {null_text}\n"
        f"Numeric stats (truncated):\n{num_stats}\n"
        f"Top categories (truncated):\n{cats_text}\n"
        f"Sample rows (CSV, first {min(max_rows, len(df))}):\n{sample_csv}"
    )

# ==== File upload ====
uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is not None:
    # Read the CSV
    df = pd.read_csv(uploaded)
    # Try to coerce numeric-looking object columns to numeric
    for c in df.columns:
        if df[c].dtype == "object":
            s = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce"
            )
            # If ≥80% of values convert, treat it as numeric
            if s.notna().mean() >= 0.8:
                df[c] = s

    # Preview
    st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20), use_container_width=True)

    # DuckDB setup
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    con.execute("CREATE OR REPLACE TABLE data AS SELECT * FROM df")
    row_count = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    st.info(f"DuckDB table 'data' is ready with {row_count} rows.")

    # =========================
    # Data-quality checks
    # =========================
    st.subheader("Data-quality checks")

    with st.expander("Uniqueness check"):
        key_col = st.selectbox("Choose a candidate key column", options=list(df.columns), key="dq_key")
        if st.button("Run uniqueness check", key="uniq_btn"):
            res = check_uniqueness(df, key_col)
            icon = "✅" if res["status"] == "PASS" else "❌"
            st.write(f"{icon} {res['check']}: **{res['status']}**")
            st.write(res["details"])

    with st.expander("Null threshold"):
        cols = st.multiselect("Choose important columns", options=list(df.columns), key="dq_null_cols")
        thr = st.number_input("Max allowed null %", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key="dq_null_thr")
        if st.button("Run null threshold check", key="null_btn"):
            res2 = check_null_threshold(df, cols, threshold_pct=thr)
            icon2 = "✅" if res2["status"] == "PASS" else "❌"
            st.write(f"{icon2} {res2['check']}: **{res2['status']}**")
            st.write(res2["details"])
            if res2.get("per_column"):
                st.dataframe(pd.DataFrame(res2["per_column"]), use_container_width=True)

    with st.expander("Value range"):
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            st.info("No numeric columns found.")
        else:
            vr_col = st.selectbox("Numeric column", options=num_cols, key="dq_range_col")
            min_allowed = st.number_input("Minimum allowed value", value=0.0, step=1.0, key="dq_range_min")
            mult = st.number_input("Max allowed as multiple of P99", value=2.0, step=0.1, key="dq_range_mult")
            if st.button("Run value range check", key="range_btn"):
                res3 = check_value_range(df, vr_col, min_allowed=min_allowed, max_multiplier_of_p99=mult)
                icon3 = "✅" if res3["status"] == "PASS" else ("⚠️" if res3["status"] == "SKIP" else "❌")
                st.write(f"{icon3} {res3['check']}: **{res3['status']}**")
                st.write(res3["details"])

    # =========================
    # NL → SQL (Ollama) + run
    # =========================
    st.subheader("Ask a question (NL → SQL)")
    question = st.text_input("Example: total revenue by category, top 5", key="nlq")

    if st.button("Generate SQL", key="gen_sql_btn"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            schema = ", ".join([f"{c} ({str(df[c].dtype)})" for c in df.columns])

            payload = {
                "model": "llama3.1",
                "messages": [
                    {"role": "system", "content": SQL_SYSTEM},
                    {"role": "user", "content": f"Schema: {schema}\nQuestion: {question}\nReturn only SQL."}
                ],
                "stream": False,
                "options": {"temperature": 0.2}
            }

            try:
                url = OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
                r = requests.post(url, json=payload, timeout=120)
                r.raise_for_status()
                sql_text = r.json().get("message", {}).get("content", "").strip()
                st.code(sql_text or "(no SQL returned)", language="sql")
            except Exception as e:
                st.error(f"Ollama request failed: {e}")
                sql_text = ""

            # Validate + execute
            try:
                if sql_text:
                    sql = validate_sql(sql_text)
                    result_df = con.execute(sql).df()
                    st.dataframe(result_df, use_container_width=True)

                    # Quick chart if 2 columns and second is numeric
                    if result_df.shape[1] == 2:
                        x_col, y_col = result_df.columns
                        if pd.api.types.is_numeric_dtype(result_df[y_col]):
                            st.bar_chart(result_df.set_index(x_col)[y_col])
            except Exception as e:
                st.error(f"SQL validation/execution failed: {e}")

    # =========================
    # Auto Insights (LLM)
    # =========================
    st.subheader("Auto Insights")

    # Show saved insights if present
    if st.session_state.get("insights_md"):
        st.markdown("**Saved insights:**")
        st.markdown(st.session_state["insights_md"])

    if st.button("Generate insights", key="insights_btn"):
        # silently pick a label/metric so Exec Summary can reuse them
        try:
            label_col, measure_col = _auto_pick_insight_columns(df)
        except Exception:
            label_col, measure_col = df.columns[0], None
        chosen_metric = measure_col if (measure_col and measure_col in df.columns and pd.api.types.is_numeric_dtype(df[measure_col])) else "(row count)"
        st.session_state["ins_label"] = label_col
        st.session_state["ins_metric"] = chosen_metric

        # build compact context for the LLM
        context = _summarize_df_for_llm(df)

        payload = {
            "model": "llama3.1",
            "messages": [
                {"role": "system", "content": INSIGHTS_SYSTEM},
                {"role": "user", "content": f"Use the dataset summary below to produce 3–5 concise, testable bullets.\n\n{context}"}
            ],
            "stream": False,
            "options": {"temperature": 0.2}
        }

        try:
            url = OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            insights_md = r.json().get("message", {}).get("content", "").strip()
            # Basic sanity: require "- " bullets; fallback note if empty
            if not insights_md or "- " not in insights_md:
                insights_md = "- Insights could not be generated with the provided context."
            st.markdown(insights_md)
            st.session_state["insights_md"] = insights_md
        except Exception as e:
            st.error(f"Ollama request failed: {e}")



    # =========================
    # Executive Summary (Ollama)
    # =========================
    st.subheader("Executive Summary")

    # Show saved summary if present
    if st.session_state.get("exec_summary"):
        st.markdown("**Saved executive summary:**")
        st.markdown(st.session_state["exec_summary"])

    if st.button("Generate executive summary", key="summary_btn"):
        # Schema & stats
        schema = ", ".join([f"{c} ({str(df[c].dtype)})" for c in df.columns])
        rows, cols = df.shape
        nulls_top = df.isna().sum().sort_values(ascending=False).head(10)
        null_text = "; ".join([f"{c}:{int(n)}" for c, n in nulls_top.items() if n > 0]) or "none"

        # Use the same selections as insights if available; otherwise pick a default
        label_col = st.session_state.get("ins_label")
        measure_col = st.session_state.get("ins_metric")

        facts_text = "No quantitative facts available."
        try:
            if label_col in df.columns:
                if measure_col == "(row count)" or measure_col not in df.columns:
                    total = int(len(df))
                    topn = df[label_col].value_counts().head(5)
                    topn_text = "; ".join([f"{str(k)}:{int(v)}" for k, v in topn.items()])
                    metric_name = "rows"
                else:
                    total = float(df[measure_col].sum())
                    topn_series = (
                        df.groupby(label_col)[measure_col]
                          .sum()
                          .sort_values(ascending=False)
                          .head(5)
                    )
                    topn_text = "; ".join([f"{str(k)}:{int(v)}" for k, v in topn_series.items()])
                    metric_name = measure_col
                facts_text = (
                    f"Total {metric_name}: {int(total)}\n"
                    f"Top {label_col} by {metric_name}: {topn_text}\n"
                    f"Unique {label_col}: {df[label_col].nunique()}"
                )
        except Exception:
            pass  # keep default facts_text

        payload = {
            "model": "llama3.1",
            "messages": [
                {"role": "system", "content": EXEC_SUMMARY_SYSTEM},
                {"role": "user", "content":
                    f"Schema: {schema}\nRows: {rows}, Cols: {cols}\nTop nulls: {null_text}\nFacts from data:\n{facts_text}\nWrite the summary now."}
            ],
            "stream": False,
            "options": {"temperature": 0.2}
        }

        try:
            url = OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            summary_text = r.json().get("message", {}).get("content", "").strip()
            st.markdown(summary_text or "(no summary returned)")
            st.session_state["exec_summary"] = summary_text
        except Exception as e:
            st.error(f"Ollama request failed: {e}")

    # =========================
    # Export report
    # =========================
    st.subheader("Export report")

    exec_summary = st.session_state.get("exec_summary")
    insights_md = st.session_state.get("insights_md")

    if not exec_summary or not insights_md:
        st.warning("Generate both the Executive Summary and Insights before exporting.")
    else:
        rows, cols = df.shape
        nulls_top = df.isna().sum().sort_values(ascending=False).head(10)
        null_text = "; ".join([f"{c}:{int(n)}" for c, n in nulls_top.items() if n > 0]) or "none"

        report_md = f"""# Data Brief: {uploaded.name}

## Dataset
Rows: {rows} | Cols: {cols}
Top nulls: {null_text}

## Executive Summary
{exec_summary}

## Insights
{insights_md}
"""
        st.download_button(
            label="Download report.md",
            data=report_md,
            file_name="report.md",
            mime="text/markdown",
        )
