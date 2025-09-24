import pandas as pd

def check_uniqueness(df: pd.DataFrame, column: str, name: str = "Uniqueness"):
    total = len(df)
    dup_count = df.duplicated(subset=[column]).sum()
    pct = (dup_count / total * 100) if total else 0.0
    status = "PASS" if dup_count == 0 else "FAIL"
    msg = f"{dup_count} duplicate rows ({pct:.2f}%) for key '{column}'"
    return {"check": name, "status": status, "details": msg}
def check_null_threshold(df, columns, threshold_pct=5.0, name="Null threshold"):
    total = len(df)
    results = []
    for col in columns:
        nulls = df[col].isna().sum()
        pct = (nulls / total * 100) if total else 0.0
        status = "PASS" if pct <= threshold_pct else "FAIL"
        results.append({"column": col, "nulls": int(nulls), "pct": round(pct, 2), "status": status})
    overall = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
    msg = " | ".join([f"{r['column']}: {r['pct']}% nulls" for r in results])
    return {"check": f"{name} ≤ {threshold_pct:.1f}%", "status": overall, "details": msg, "per_column": results}
def check_value_range(df, column, min_allowed=0.0, max_multiplier_of_p99=2.0, name="Value range"):
    import pandas as pd
    if column not in df.columns:
        return {"check": name, "status": "FAIL", "details": f"Column '{column}' not found"}

    ser = df[column].dropna()
    if ser.empty:
        return {"check": name, "status": "PASS", "details": f"No values present in '{column}'"}

    if not pd.api.types.is_numeric_dtype(ser):
        return {"check": name, "status": "SKIP", "details": f"Column '{column}' is not numeric"}

    p99 = float(ser.quantile(0.99))
    max_allowed = p99 * float(max_multiplier_of_p99)

    below_min = int((ser < min_allowed).sum())
    above_max = int((ser > max_allowed).sum())

    status = "PASS" if below_min == 0 and above_max == 0 else "FAIL"
    details = (
        f"min_allowed={min_allowed}, max_allowed≈{max_allowed:.2f} (≈{max_multiplier_of_p99}×P99={p99:.2f}); "
        f"{below_min} below min, {above_max} above max in '{column}'"
    )
    return {"check": name, "status": status, "details": details}

