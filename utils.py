# --- utils.py ---
# Utilities for the Dubai AI Shopping Assistant dashboard
# Safe defaults + helpers for composites, coefficients, predictions, and Streamlit session state.

from __future__ import annotations
import math
import re
from typing import Dict, Any, Optional, List

import pandas as pd

# Optional: Streamlit access (gracefully degrade if not in a Streamlit context)
try:
    import streamlit as st
except Exception:  # pragma: no cover
    class _Stub:
        session_state = {}
        def warning(self, *a, **k): pass
        def info(self, *a, **k): pass
        def error(self, *a, **k): pass
        # Compat shims for query params (older/newer Streamlit)
        @property
        def query_params(self):  # type: ignore
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

# Adjust these strings only if your sheet headers have changed materially.
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

def _coerce_map(schema: Optional[Dict[str, Any]]) -> Dict[Any, int]:
    """Return a Likert map from schema or fall back to defaults."""
    if schema and isinstance(schema, dict) and "likert_map" in schema:
        lm = dict(DEFAULT_LIKERT_MAP)  # ensure numeric keys exist too
        lm.update(schema["likert_map"])
        return lm
    return DEFAULT_LIKERT_MAP

def _normalize_label(s: str) -> str:
    return re.sub(r"\W+", "", s.strip().lower())

def _soft_match(probe: str, candidate: str, prefix: int = 20) -> bool:
    """Soft match to tolerate small column name changes."""
    return _normalize_label(probe)[:prefix] in _normalize_label(candidate)

def _composite_schema(df: pd.DataFrame, schema: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Build a composite schema using provided schema if present, else defaults.
    Drops columns not found in df. Attempts soft-matching when needed.
    """
    base = (schema.get("composites") if schema and "composites" in schema else DEFAULT_COMPOSITE_SCHEMA)
    resolved: Dict[str, List[str]] = {}
    for k, cols in base.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            # Try soft matches if exact headers changed
            for c in df.columns:
                if any(_soft_match(probe, c) for probe in cols):
                    present.append(c)
        # De-dupe while preserving order
        resolved[k] = list(dict.fromkeys(present))
    return resolved


# --------------------------------------------
# 2) Likert normalization and composite scores
# --------------------------------------------

def normalize_likert(df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Map Likert text → numeric safely, leaving non-Likert columns unchanged.
    - If schema has 'likert_map', use it on top of defaults; else use defaults.
    """
    lm = _coerce_map(schema)
    out = df.copy()
    for col in out.columns:
        # Try mapping object columns or any column containing keys from the map
        try_map = False
        if out[col].dtype == object:
            try_map = True
        else:
            # If any value is in the map keys, we can apply it
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
    """
    Return a DataFrame with composite scores for each construct.
    If a construct has no matching columns, it returns NaNs (handled upstream).
    """
    df_num = normalize_likert(df, schema)
    comp_schema = _composite_schema(df, schema)
    out: Dict[str, pd.Series] = {}
    for construct, cols in comp_schema.items():
        if cols:
            out[construct] = pd.to_numeric(df_num[cols], errors="coerce").mean(axis=1)
        else:
            out[construct] = pd.Series([pd.NA] * len(df_num), index=df_num.index)
    return pd.DataFrame(out)


# ---------------------------------------
# 3) Coefficients / segments: load & save
# ---------------------------------------

def get_models_from_session():
    """
    Return coefficient dicts (coeffs) and segments from st.session_state, if any.
    Expected keys:
      - 'coeffs_intent' (dict)  and 'coeffs_choice' (dict) optionally combined under 'coeffs'
      - 'segments' (dict)
    """
    coeffs = st.session_state.get("coeffs")  # combined form {'intent': {...}, 'choice': {...}}
    coeffs_intent = st.session_state.get("coeffs_intent")
    coeffs_choice = st.session_state.get("coeffs_choice")
    segs = st.session_state.get("segments")
    # Normalize to a single dict
    if not coeffs:
        coeffs = {}
        if coeffs_intent: coeffs["intent"] = coeffs_intent
        if coeffs_choice: coeffs["choice"] = coeffs_choice
    return coeffs or None, segs or None

def load_coeffs() -> Dict[str, Any]:
    """
    Load coefficients for the simulator.
    If none in session, return conservative 'evidence-mode' defaults consistent with your findings.
    - 'intent' is a linear model on 1–5 scale.
    - 'choice' is a logistic model (probability via sigmoid).
    """
    coeffs_sess, _ = get_models_from_session()
    if coeffs_sess:
        return coeffs_sess

    # Evidence-mode defaults (direction/magnitude aligned with N=94 narrative):
    coeffs = {
        "intent": {
            "intercept": 1.00,
            "Usefulness": 0.12,
            "EaseOfUse": 0.05,              # small direct effect on intent
            "Trust": 0.38,                   # largest
            "CulturalFit": 0.28,             # second largest
            "EmotionalConnection": 0.14,
            "TechComfort": 0.06
        },
        "choice": {
            "intercept": -1.20,              # baseline odds toward Human
            "Usefulness": 0.05,              # control-sized
            "EaseOfUse": 0.28,               # adds lift
            "Trust": 0.22,                   # adds lift
            "CulturalFit": 0.06,             # small direct in choice
            "EmotionalConnection": 0.55,     # biggest mover
            "TechComfort": 0.04
        },
        "meta": {
            "type": "evidence-defaults",
            "notes": "Replace by recomputing models in Methods. Values reflect N=94 story: Trust/Culture→Intent; Emotion(+Ease,Trust)→Choice."
        }
    }
    return coeffs

def load_segments() -> Dict[str, Any]:
    """
    Load segment presets. If none in session, return reasonable means for three classes.
    """
    _, segs_sess = get_models_from_session()
    if segs_sess:
        return segs_sess

    segments = {
        "segments": [
            {
                "name": "Digitally Native Loyalists",
                "share": 0.22,
                "means": {
                    "Usefulness": 4.4, "EaseOfUse": 4.5, "Trust": 4.3,
                    "CulturalFit": 4.2, "EmotionalConnection": 4.4, "TechComfort": 4.6
                }
            },
            {
                "name": "Pragmatic Adopters",
                "share": 0.46,
                "means": {
                    "Usefulness": 4.0, "EaseOfUse": 4.1, "Trust": 3.2,
                    "CulturalFit": 3.6, "EmotionalConnection": 3.5, "TechComfort": 4.2
                }
            },
            {
                "name": "Cautious Minimalists",
                "share": 0.32,
                "means": {
                    "Usefulness": 3.6, "EaseOfUse": 3.5, "Trust": 2.9,
                    "CulturalFit": 3.1, "EmotionalConnection": 3.0, "TechComfort": 3.9
                }
            },
        ],
        "meta": {"type": "presets", "notes": "Replace with LCA means if available."}
    }
    return segments


# --------------------------------------
# 4) Session data access (uploaded file)
# --------------------------------------

def get_data_from_session() -> Optional[pd.DataFrame]:
    """
    Return the uploaded/active DataFrame if your app stored it in session_state['df'].
    Adjust this if you use a different key.
    """
    df = st.session_state.get("df")
    try:
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            return df
    except Exception:
        pass
    return None


# -----------------------
# 5) Prediction functions
# -----------------------

def _linear_predict(x: Dict[str, float], w: Dict[str, float]) -> float:
    y = w.get("intercept", 0.0)
    for k, v in x.items():
        if k in w:
            y += float(w[k]) * float(v)
    return y

def _sigmoid(z: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:  # pragma: no cover
        return 0.0 if z < 0 else 1.0

def predict_intent(constructs: Dict[str, float], coeffs: Dict[str, Any]) -> float:
    """
    Predict willingness (1–5). Linear combination, then clamp to [1, 5].
    """
    w = (coeffs.get("intent") if coeffs else None) or load_coeffs().get("intent")
    score = _linear_predict(constructs, w)
    # Clamp within Likert bounds
    return max(1.0, min(5.0, float(score)))

def predict_choice_probability(constructs: Dict[str, float], coeffs: Dict[str, Any]) -> float:
    """
    Predict P(choose AI) via logistic regression.
    """
    w = (coeffs.get("choice") if coeffs else None) or load_coeffs().get("choice")
    z = _linear_predict(constructs, w)
    return float(_sigmoid(z))


# -----------------------------------
# 6) URL query param helpers (Streamlit)
# -----------------------------------

def get_query_params() -> Dict[str, List[str]]:
    """Read query params, compatible with older/newer Streamlit."""
    try:
        return st.query_params.to_dict()  # type: ignore[attr-defined]
    except Exception:
        try:
            return st.experimental_get_query_params()  # type: ignore[attr-defined]
        except Exception:
            return {}

def set_query_params(d: Dict[str, Any]) -> None:
    """Write query params, compatible with older/newer Streamlit."""
    # cast to str for URL
    payload = {k: str(v) for k, v in d.items()}
    try:
        st.query_params.update(payload)  # type: ignore[attr-defined]
    except Exception:
        try:
            st.experimental_set_query_params(**payload)  # type: ignore[attr-defined]
        except Exception:
            pass


# -----------------------------------
# 7) Convenience: baseline composites
# -----------------------------------

def composites_from_df_means(df: Optional[pd.DataFrame], schema: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Compute mean composite values from a DataFrame, falling back to sensible defaults.
    """
    defaults = {
        "Usefulness": 3.5, "EaseOfUse": 3.5, "Trust": 3.2,
        "CulturalFit": 3.4, "EmotionalConnection": 3.2, "TechComfort": 4.0
    }
    if df is None or len(df) == 0:
        return defaults
    try:
        comp = compute_composites(df, schema)
        if comp is None or comp.empty:
            return defaults
        means = comp.mean(numeric_only=True).to_dict()
        out = {}
        for k, v in defaults.items():
            val = means.get(k, float('nan'))
            try:
                out[k] = float(round(val, 2)) if pd.notna(val) else v
            except Exception:
                out[k] = v
        return out
    except Exception:
        return defaults
