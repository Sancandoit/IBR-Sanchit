# --- utils.py ---
# Utilities for the Dubai AI Shopping Assistant dashboard

from __future__ import annotations
import math, re, json, os, zipfile, io
from typing import Dict, Any, Optional, List
import pandas as pd

# Optional: Streamlit access
try:
    import streamlit as st
except Exception:  # fallback stub for non-Streamlit context
    class _Stub:
        session_state = {}
        def warning(self, *a, **k): print("Warning:", a, k)
        def error(self, *a, **k): print("Error:", a, k)
        def info(self, *a, **k): print("Info:", a, k)
        @property
        def query_params(self): 
            class _QP:
                def to_dict(self): return {}
                def update(self, *a, **k): pass
            return _QP()
        def experimental_get_query_params(self): return {}
        def experimental_set_query_params(self, **kwargs): pass
    st = _Stub()  # type: ignore

# -----------------------------
# 1) Defaults and schema helpers
# -----------------------------
DEFAULT_LIKERT_MAP: Dict[Any, int] = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5
}

DEFAULT_COMPOSITE_SCHEMA: Dict[str, List[str]] = {
    "Usefulness": [
        "AI assistants help me make better shopping decisions.",
        "AI assistants save me time in-store or online."
    ],
    "EaseOfUse": [
        "I find AI shopping assistants easy to use and understand.",
        "It doesn’t take much effort to learn how to use these assistants."
    ],
    "Trust": [
        "I trust recommendations made by AI shopping assistants.",
        "I feel confident that my data is safe when using AI features."
    ],
    "CulturalFit": [
        "I appreciate when an AI shopping assistant speaks my language or uses familiar cultural references.",
        "The tone and style of AI assistants in Dubai suit me."
    ],
    "EmotionalConnection": [
        "I enjoy chatting with an AI shopping assistant.",
        "Sometimes, AI assistants “get me” better than human staff."
    ],
    "TechComfort": [
        "How comfortable are you with using new digital technology?"
    ],
}

DEFAULT_DEMOGRAPHICS: Dict[str, str] = {
    "age": "How old are you?",
    "nationality": "What best describes you?",
    "shopping": "How do you usually shop in Dubai?",
    "tech_comfort": "How comfortable are you with using new digital technology?"
}

def _coerce_map(schema: Optional[Dict[str, Any]]) -> Dict[Any, int]:
    if schema and isinstance(schema, dict) and "likert_map" in schema:
        lm = dict(DEFAULT_LIKERT_MAP)
        lm.update(schema["likert_map"])
        return lm
    return DEFAULT_LIKERT_MAP

def _normalize_label(s: str) -> str:
    return re.sub(r"\W+", "", s.strip().lower())

def _soft_match(probe: str, candidate: str, prefix: int = 20) -> bool:
    return _normalize_label(probe)[:prefix] in _normalize_label(candidate)

def _composite_schema(df: pd.DataFrame, schema: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    base = (schema.get("composites") if schema and "composites" in schema else DEFAULT_COMPOSITE_SCHEMA)
    resolved: Dict[str, List[str]] = {}
    for k, cols in base.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            for c in df.columns:
                if any(_soft_match(probe, c) for probe in cols):
                    present.append(c)
        resolved[k] = list(dict.fromkeys(present))
    return resolved

# --------------------------------------------
# 2) Likert normalization and composite scores
# --------------------------------------------
def normalize_likert(df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    lm = _coerce_map(schema)
    out = df.copy()
    for col in out.columns:
        try_map = False
        if out[col].dtype == object:
            try_map = True
        else:
            try:
                vals = out[col].dropna().unique()
                if any(v in lm for v in vals):
                    try_map = True
            except Exception:
                pass
        if try_map:
            mapped = out[col].map(lm)
            out[col] = mapped.where(mapped.notna(), out[col])
    return out

def compute_composites(df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    df_num = normalize_likert(df, schema)
    comp_schema = _composite_schema(df, schema)
    out: Dict[str, pd.Series] = {}
    for construct, cols in comp_schema.items():
        if not cols:
            out[construct] = pd.Series([pd.NA] * len(df_num), index=df_num.index)
            continue
        if len(cols) == 1:
            out[construct] = pd.to_numeric(df_num[cols[0]], errors="coerce")
        else:
            sub = df_num[cols].apply(pd.to_numeric, errors="coerce")
            out[construct] = sub.mean(axis=1)
    return pd.DataFrame(out)

# ---------------------------------------
# 3) Coefficients / segments: load & save
# ---------------------------------------
def get_models_from_session():
    """
    Safely retrieve coefficients and segments from session state.
    """
    coeffs = st.session_state.get("coeffs", None)
    segs = st.session_state.get("segments", None)
    return coeffs, segs

def load_coeffs() -> Dict[str, Any]:
    """
    Load coefficients either from session or fall back to defaults.
    """
    coeffs_sess, _ = get_models_from_session()
    if isinstance(coeffs_sess, dict) and coeffs_sess:
        return coeffs_sess

    # Fallback defaults
    return {
        "intent": {
            "intercept": 1.00,
            "Usefulness": 0.12,
            "EaseOfUse": 0.05,
            "Trust": 0.38,
            "CulturalFit": 0.28,
            "EmotionalConnection": 0.14,
            "TechComfort": 0.06
        },
        "choice": {
            "intercept": -1.20,
            "Usefulness": 0.05,
            "EaseOfUse": 0.28,
            "Trust": 0.22,
            "CulturalFit": 0.06,
            "EmotionalConnection": 0.55,
            "TechComfort": 0.04
        },
        "meta": {
            "type": "evidence-defaults",
            "notes": "Replace by recomputing models in Methods."
        }
    }

def load_segments() -> Dict[str, Any]:
    _, segs_sess = get_models_from_session()
    if isinstance(segs_sess, dict) and segs_sess:
        return segs_sess
    return {
        "segments": [
            {"name": "Digitally Native Loyalists", "share": 0.22,
             "means": {"Usefulness": 4.4, "EaseOfUse": 4.5, "Trust": 4.3,
                       "CulturalFit": 4.2, "EmotionalConnection": 4.4, "TechComfort": 4.6}},
            {"name": "Pragmatic Adopters", "share": 0.46,
             "means": {"Usefulness": 4.0, "EaseOfUse": 4.1, "Trust": 3.2,
                       "CulturalFit": 3.6, "EmotionalConnection": 3.5, "TechComfort": 4.2}},
            {"name": "Cautious Minimalists", "share": 0.32,
             "means": {"Usefulness": 3.6, "EaseOfUse": 3.5, "Trust": 2.9,
                       "CulturalFit": 3.1, "EmotionalConnection": 3.0, "TechComfort": 3.9}}
        ],
        "meta": {"type": "presets", "notes": "Replace with LCA means if available."}
    }

# --------------------------------------
# 4) Session data access (uploaded file)
# --------------------------------------
def get_data_from_session() -> Optional[pd.DataFrame]:
    df = st.session_state.get("df")
    try:
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            return df
    except Exception:
        pass
    return None

# --------------------------------------
# File Reading Utility (safe)
# --------------------------------------
def _read_any(file) -> pd.DataFrame:
    """
    Read a file (CSV/XLSX) safely into a DataFrame.
    Used by app.py for uploading survey responses.
    """
    if file is None:
        return pd.DataFrame()

    name = getattr(file, "name", None) or str(file)

    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Could not read file {name}: {e}")
        return pd.DataFrame()

    return pd.DataFrame()

def set_data_in_session(df: pd.DataFrame) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        return
    st.session_state["df"] = df

# -----------------------
# 5) Prediction functions
# -----------------------
def _linear_predict(constructs: dict, weights: dict):
    if not isinstance(weights, dict) or not weights:
        return 0.0
    y = weights.get("intercept", 0.0)
    for k, v in constructs.items():
        coef = weights.get(k, 0.0)
        try:
            y += coef * float(v)
        except (ValueError, TypeError):
            continue
    return y

def predict_intent(constructs: dict, coeffs: dict):
    if not coeffs or not isinstance(coeffs, dict):
        coeffs = load_coeffs()
    w = coeffs.get("intent", coeffs)
    return _linear_predict(constructs, w)

def _sigmoid(z: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z < 0 else 1.0

def predict_choice_probability(constructs: Dict[str, float], coeffs: Dict[str, Any]) -> float:
    if not coeffs or not isinstance(coeffs, dict):
        coeffs = load_coeffs()
    w = coeffs.get("choice", coeffs)
    z = _linear_predict(constructs, w)
    return float(_sigmoid(z))

# -----------------------------------
# 6) URL query param helpers
# -----------------------------------
def get_query_params() -> Dict[str, List[str]]:
    try:
        return st.query_params.to_dict()  # type: ignore
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}

def set_query_params(d: Dict[str, Any]) -> None:
    payload = {k: str(v) for k, v in d.items()}
    try:
        st.query_params.update(payload)  # type: ignore
    except Exception:
        try: st.experimental_set_query_params(**payload)
        except Exception: pass

# -----------------------------------
# 7) Convenience: baseline composites
# -----------------------------------
def composites_from_df_means(df: Optional[pd.DataFrame], schema: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    defaults = {"Usefulness": 3.5, "EaseOfUse": 3.5, "Trust": 3.2,
                "CulturalFit": 3.4, "EmotionalConnection": 3.2, "TechComfort": 4.0}
    if df is None or len(df) == 0: return defaults
    try:
        comp = compute_composites(df, schema)
        if comp is None or comp.empty: return defaults
        means = comp.mean(numeric_only=True).to_dict()
        out = {}
        for k, v in defaults.items():
            val = means.get(k, float('nan'))
            try: out[k] = float(round(val, 2)) if pd.notna(val) else v
            except Exception: out[k] = v
        return out
    except Exception: return defaults

# -----------------------------------
# 8) Methods page helpers
# -----------------------------------
def load_schema(path: Optional[str] = None) -> Dict[str, Any]:
    if not path:
        return {
            "likert_map": DEFAULT_LIKERT_MAP,
            "composites": DEFAULT_COMPOSITE_SCHEMA,
            "constructs": DEFAULT_COMPOSITE_SCHEMA,
            "demographics": DEFAULT_DEMOGRAPHICS,
            "age": DEFAULT_DEMOGRAPHICS["age"],
            "nationality": DEFAULT_DEMOGRAPHICS["nationality"],
            "shopping": DEFAULT_DEMOGRAPHICS["shopping"],
            "tech_comfort": DEFAULT_DEMOGRAPHICS["tech_comfort"]
        }
    try:
        if path.endswith(".json"):
            with open(path, "r") as f:
                schema = json.load(f)
        elif path.endswith((".yml", ".yaml")):
            import yaml
            with open(path, "r") as f:
                schema = yaml.safe_load(f)
        else:
            schema = {}
    except Exception as e:
        st.warning(f"Could not load schema {path}: {e}")
        schema = {}
    schema.setdefault("likert_map", DEFAULT_LIKERT_MAP)
    schema.setdefault("composites", DEFAULT_COMPOSITE_SCHEMA)
    schema.setdefault("constructs", schema.get("composites", DEFAULT_COMPOSITE_SCHEMA))
    schema.setdefault("demographics", DEFAULT_DEMOGRAPHICS)
    for k, v in DEFAULT_DEMOGRAPHICS.items():
        schema.setdefault(k, v)
    return schema

def reliability_table(df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    from math import isnan
    df_num = normalize_likert(df, schema)
    comp_schema = _composite_schema(df, schema)
    rows = []
    for construct, cols in comp_schema.items():
        if len(cols) < 2: continue
        sub = df_num[cols].apply(pd.to_numeric, errors="coerce")
        k = sub.shape[1]
        var_sum = sub.var(axis=0, ddof=1).sum()
        total_var = sub.sum(axis=1).var(ddof=1)
        alpha = (k / (k - 1)) * (1 - var_sum / total_var) if total_var > 0 else float("nan")
        rows.append({"Construct": construct, "Items": k,
                     "CronbachAlpha": round(alpha, 3) if not isnan(alpha) else None})
    return pd.DataFrame(rows)

def set_models_in_session(coeffs: Dict[str, Any], segments: Dict[str, Any]) -> None:
    if coeffs: st.session_state["coeffs"] = coeffs
    if segments: st.session_state["segments"] = segments

def make_snapshot_zip(schema: Dict[str, Any],
                      coeffs: Dict[str, Any],
                      segments: Dict[str, Any],
                      df: pd.DataFrame,
                      out_path: str = "snapshot.zip") -> str:
    with zipfile.ZipFile(out_path, "w") as zf:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        zf.writestr("data.csv", buf.getvalue())
        zf.writestr("coeffs.json", json.dumps(coeffs, indent=2))
        zf.writestr("segments.json", json.dumps(segments, indent=2))
        if schema:
            zf.writestr("schema.json", json.dumps(schema, indent=2))
    return out_path
