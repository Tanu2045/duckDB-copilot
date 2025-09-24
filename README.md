# DuckDB Copilot — NL→SQL + Data Quality + Auto-Report

A lightweight analyst copilot that combines natural language to SQL, automated data-quality checks, and auto-generated reports from any dataset.

Runs locally in your browser with Streamlit.


## What it does
- Upload CSV/Parquet → loads into DuckDB (fast, zero setup, trending in analytics).
- Ask in plain English → app uses an LLM to generate safe SELECT-only SQL → executes in DuckDB → shows table + quick chart.
- Run data-quality checks → a minimal suite runs:
  - Uniqueness of key columns  
  - Null thresholds  
  - Value ranges for numeric columns  
  → outputs simple PASS/FAIL results.
- Generate report → creates an executive summary with:
  - Insights bullets  
  - Validation summary  
  - Saved as clean Markdown (report.md) you could hand to a manager.


## Stack
- Python + Streamlit for UI
- DuckDB for analytics
- Ollama (local LLM) for NL→SQL — no API key needed
- Lightweight, homegrown DQ checks 

