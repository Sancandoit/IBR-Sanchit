import json, re, io, zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

# ---------- Session helpers ----------
def set_data_in_session(df: pd.DataFrame):
    st.session_state["raw_df"] = df

def get_data_from_session():
    return st.session_state.get("raw_df")

def set_models_in_session(coeffs: dict, segs: dict):
    st.session_state["coeffs"] = coeffs
    st.session_state["segments"] = segs

def get_models_from_session():
    return st.session_state.get("coeffs"), st.session_state.get("segments")

# ---------- File / config loaders ----------
@st.cache_data(show_spinner=False)
def load_schema():
    return json.load(open(Path("config")/"schema.json", encoding="utf-8"))

@st.cache_data(show_spinner=False)
def load_coeffs():
    return json.load(open(Path("models")/"coeffs.json", encoding="utf-8"))

@st.cache_data(show_spinner=False)
def load_segments():
    return json.load(open(Path("models")/"segments.json", encoding="utf-8"))

@st.cache_data(show_spinner=False)
def _read_any(file):
    # file can be UploadedFile or a filesystem path
    if hasattr(file, "name"):
        name = file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    p = Path(file)
    if p.suffix.lower()==".csv":
        return pd.read_csv(p)
    return pd.read_excel(p)

# ---------- Data wrangling ----------
def _to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def normalize_likert(df, schema):
    lm = schema["likert_map"]
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].map(lm).fillna(out[c])
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out

def find_col(df, pattern, case=False):
    flags = 0 if case else re.I
    for c in df.columns:
        if re.search(pattern, str(c), flags):
            return c
    return None

def compute_composites(df, schema):
    df_num = normalize_likert(df, schema)
    composites = {}
    for k, cols in schema["constructs"].items():
        # match by prefix of each item text to allow minor wording differences
        cols_present = [c for c in df_num.columns if any(cc[:15].lower() in c.lower() for cc in cols)]
        if cols_present:
            composites[k] = df_num[cols_present].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    tc_col = find_col(df_num, f'^{re.escape(schema["tech_comfort"][:15])}')
    if tc_col:
        composites["TechComfort"] = pd.to_numeric(df_num[tc_col], errors="coerce")
    return pd.DataFrame(composites)

# ---------- Reliability ----------
def cronbach_alpha(df_sub: pd.DataFrame):
    s = df_sub.dropna(axis=0, how="any")
    if s.shape[1] < 2 or s.shape[0] < 3:
        return np.nan
    k = s.shape[1]
    var_sum = s.var(axis=0, ddof=1).sum()
    total_var = s.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k/(k-1)) * (1 - var_sum/total_var)

def reliability_table(df, schema):
    rows = []
    df_num = normalize_likert(df, schema)
    for k, cols in schema["constructs"].items():
        cols_present = [c for c in df_num.columns if any(cc[:15].lower() in c.lower() for cc in cols)]
        if len(cols_present) >= 2:
            alpha = cronbach_alpha(df_num[cols_present].apply(pd.to_numeric, errors="coerce"))
            rows.append({"Construct": k, "Items": len(cols_present), "Alpha": round(alpha,3)})
        else:
            rows.append({"Construct": k, "Items": len(cols_present), "Alpha": None})
    return pd.DataFrame(rows)

# ---------- Simple predictors (for simulator) ----------
def predict_intent(comp, coeffs):
    b = coeffs["intent_ols"]
    y = (b["intercept"]
         + b.get("Usefulness",0)*comp.get("Usefulness",0)
         + b.get("EaseOfUse",0)*comp.get("EaseOfUse",0)
         + b.get("Trust",0)*comp.get("Trust",0)
         + b.get("CulturalFit",0)*comp.get("CulturalFit",0)
         + b.get("EmotionalConnection",0)*comp.get("EmotionalConnection",0)
         + b.get("TechComfort",0)*comp.get("TechComfort",0))
    return float(max(1.0, min(5.0, y)))

def predict_choice_probability(comp, coeffs):
    b = coeffs["choice_logit"]
    z = (b["intercept"]
         + b.get("EmotionalConnection",0)*comp.get("EmotionalConnection",0)
         + b.get("EaseOfUse",0)*comp.get("EaseOfUse",0)
         + b.get("Trust",0)*comp.get("Trust",0))
    return float(1.0/(1.0+np.exp(-z)))

# ---------- Reproducibility snapshot ----------
def make_snapshot_zip(schema:dict, coeffs:dict, segs:dict, df_sample:pd.DataFrame):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("config/schema.json", json.dumps(schema, indent=2, ensure_ascii=False))
        z.writestr("models/coeffs.json", json.dumps(coeffs, indent=2))
        z.writestr("models/segments.json", json.dumps(segs, indent=2))
        z.writestr("data/sample_anonymized.csv", df_sample.to_csv(index=False))
    buf.seek(0)
    return buf

# ---- URL query param helpers (Streamlit v1.30+) ----
def get_query_params():
    try:
        return st.query_params.to_dict()
    except Exception:
        # Fallback for older versions
        return st.experimental_get_query_params()

def set_query_params(d: dict):
    try:
        st.query_params.update({k: str(v) for k, v in d.items()})
    except Exception:
        st.experimental_set_query_params(**{k: str(v) for k, v in d.items()})
