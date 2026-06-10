"""
app_new.py — CVD Тәуекел Болжаушы · Streamlit (3 беттік нұсқа)
================================================================
Іске қосу:
    pip install streamlit shap plotly
    streamlit run app_new.py

Беттер:
  1. warning  — өлшеу алдындағы ескертулер
  2. form     — пациент деректерін енгізу
  3. result   — болжам + клиникалық кеңестер + SHAP
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle, os, warnings
import plotly.graph_objects as go
import shap
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="CVD Risk Analyzer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _find_model_dir():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    candidates = [
        os.path.join(here, 'cvd_models'),
        os.path.join(here, '..', 'cvd_models'),
        os.getcwd(),
        os.path.join(os.getcwd(), 'cvd_models'),
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.isdir(p) and any(f.endswith('.pkl') for f in os.listdir(p)):
            return p
    return 'cvd_models'

MODEL_DIR = _find_model_dir()

# ══════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════
if 'page' not in st.session_state:
    st.session_state.page = 'warning'
if 'result_data' not in st.session_state:
    st.session_state.result_data = None

# ══════════════════════════════════════════════════════
# SBP / DBP КАТЕГОРИЯ
# ══════════════════════════════════════════════════════
def sbp_to_category(sbp_value):
    if pd.isna(sbp_value): return np.nan
    elif sbp_value <= 150: return 0
    elif sbp_value <= 170: return 1
    else: return 2

def dbp_to_category(dbp_value):
    if pd.isna(dbp_value): return np.nan
    elif dbp_value < 100: return 0
    elif dbp_value <= 120: return 1
    else: return 2

# ══════════════════════════════════════════════════════
# ЖАЛПЫ СТИЛЬ
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
  h1, h2, h3, h4, h5, h6 { font-family: 'DM Serif Display', serif !important; color: #111128 !important; }
  p, div, span, li, td, th, label { color: #7f7f96 !important; }
  .stApp { background-color: #f4f6fb !important; }

  .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
  [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
  [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
  [data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] strong,
  [data-testid="stMarkdownContainer"] b { color: #111128 !important; }

  label, .stNumberInput > label, .stSelectbox > label, .stSlider > label,
  .stToggle > label, [data-testid="stFormLabel"] label,
  [data-baseweb="form-control-label"], [data-baseweb="form-control-label"] div,
  [data-baseweb="form-control-label"] span {
      color: #111128 !important; font-size: 14px !important; font-weight: 600 !important;
  }
  .stNumberInput input {
      color: #111128 !important; background: #ffffff !important;
      border: 1px solid #c0c4d8 !important; border-radius: 8px !important; font-weight: 500 !important;
  }
  .stSelectbox > div > div, [data-baseweb="select"] span, [data-baseweb="select"] div {
      color: #111128 !important; background: #ffffff !important; border-color: #c0c4d8 !important;
  }
  [data-testid="stSlider"] label, [data-testid="stSlider"] p, .stSlider label {
      color: #111128 !important; font-weight: 700 !important; font-size: 14px !important;
  }
  [data-testid="stToggle"] label, [data-testid="stToggle"] p, .stToggle label {
      color: #111128 !important; font-weight: 700 !important; font-size: 14px !important;
  }
  .stCaption, [data-testid="stCaptionContainer"] p { color: #5a5a7a !important; }

  [data-testid="stInfo"] {
      background: #eef2ff !important; border: 1px solid #b0bcee !important; border-radius: 10px !important;
  }
  [data-testid="stInfo"] p, [data-testid="stInfo"] div, [data-testid="stInfo"] span,
  [data-testid="stInfo"] strong { color: #111128 !important; }
  [data-testid="stWarning"] { background: #fff8e6 !important; border: 1px solid #e0c060 !important; }
  [data-testid="stWarning"] p, [data-testid="stWarning"] div, [data-testid="stWarning"] span { color: #111128 !important; }
  [data-testid="stError"] p, [data-testid="stError"] div { color: #111128 !important; }
  hr { border-color: #d0d4e8 !important; }

  section[data-testid="stSidebar"] { background: #1a1d35 !important; border-right: 1px solid #2a2d50 !important; }
  section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] li { color: #dde3f5 !important; }
  section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 { color: #f0f4ff !important; }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] [data-baseweb="form-control-label"] div,
  section[data-testid="stSidebar"] [data-baseweb="form-control-label"] span {
      color: #c8d4f5 !important; font-weight: 700 !important;
  }
  section[data-testid="stSidebar"] .stSelectbox > div > div,
  section[data-testid="stSidebar"] [data-baseweb="select"] span,
  section[data-testid="stSidebar"] [data-baseweb="select"] div {
      background: #252845 !important; color: #f0f4ff !important; border-color: #3a3f70 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSlider"] label,
  section[data-testid="stSidebar"] [data-testid="stSlider"] p { color: #c8d4f5 !important; font-weight: 700 !important; }
  section[data-testid="stSidebar"] .stMarkdown p { color: #b8c8e8 !important; }

  [data-testid="stForm"] {
      background: #ffffff !important; border: 1px solid #d0d4e8 !important;
      border-radius: 16px !important; padding: 24px !important;
  }

  .card {
      background: #ffffff; border: 1px solid #d0d4e8; border-radius: 16px;
      padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 12px rgba(17,17,40,0.06);
  }
  .card-header {
      font-size: 12px !important; font-weight: 700 !important; letter-spacing: 2px;
      text-transform: uppercase; color: #5a5a7a !important; margin-bottom: 8px;
  }
  .card-value { font-size: 38px; font-weight: 700; font-family: 'DM Serif Display', serif; }

  .result-healthy {
      background: linear-gradient(135deg, #f0fff6, #e0faea);
      border: 2px solid #2a9a5a; border-radius: 20px; padding: 32px; text-align: center;
  }
  .result-sick {
      background: linear-gradient(135deg, #fff0f0, #fae0e0);
      border: 2px solid #cc3333; border-radius: 20px; padding: 32px; text-align: center;
  }

  /* НЕГІЗГІ КНОПКА (көк) */
  .stButton > button {
      background: linear-gradient(135deg, #2a4dd0, #4a6de8) !important;
      color: #ffffff !important; border: none !important;
      border-radius: 12px !important; padding: 14px 32px !important;
      font-size: 16px !important; font-weight: 700 !important;
      width: 100% !important; transition: all 0.25s ease !important;
      box-shadow: 0 4px 15px rgba(42,77,208,0.30) !important;
      letter-spacing: 0.3px !important;
  }
  .stButton > button:hover {
      transform: translateY(-2px) !important;
      box-shadow: 0 8px 25px rgba(42,77,208,0.45) !important;
  }
  .stButton > button p, .stButton > button span, .stButton > button div { color: #ffffff !important; }

  /* ЖАСЫЛ КНОПКА */
  .btn-green .stButton > button {
      background: linear-gradient(135deg, #1a7a4a, #27ae72) !important;
      box-shadow: 0 4px 20px rgba(26,122,74,0.40) !important;
      font-size: 18px !important; padding: 18px 40px !important;
  }
  .btn-green .stButton > button:hover {
      box-shadow: 0 8px 32px rgba(26,122,74,0.55) !important;
  }

  /* СҰР КНОПКА */
  .btn-grey .stButton > button {
      background: linear-gradient(135deg, #5a5a7a, #7a7a9a) !important;
      box-shadow: 0 4px 12px rgba(90,90,122,0.25) !important;
  }

  [data-testid="stDownloadButton"] button {
      background: linear-gradient(135deg, #1a6a9a, #1e82be) !important;
      color: #ffffff !important; border: none !important;
      border-radius: 10px !important; font-weight: 700 !important;
  }
  [data-testid="stDownloadButton"] button p,
  [data-testid="stDownloadButton"] button span,
  [data-testid="stDownloadButton"] button div { color: #ffffff !important; }

  [data-testid="stDataFrame"] th { color: #111128 !important; background: #eef0fa !important; }
  [data-testid="stDataFrame"] td { color: #111128 !important; }

  .warning-card {
      background: #ffffff; border: 1.5px solid #e8e0f5; border-radius: 16px;
      padding: 20px 24px; margin-bottom: 14px; box-shadow: 0 2px 10px rgba(17,17,40,0.05);
      display: flex; align-items: flex-start; gap: 14px;
  }
  .warning-icon { font-size: 28px; flex-shrink: 0; margin-top: 2px; }
  .warning-title { font-size: 15px; font-weight: 700; color: #111128 !important; margin-bottom: 4px; }
  .warning-text { font-size: 13px; color: #3a3a5c !important; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# МОДЕЛЬ ЖҮКТЕУ
# ══════════════════════════════════════════════════════
def _safe_load_pkl(path: str):
    try:
        with open(path, 'rb') as f: return pickle.load(f)
    except Exception: pass
    try:
        with open(path, 'rb') as f: return pickle.load(f, encoding='latin1')
    except Exception: pass
    try:
        import joblib; return joblib.load(path)
    except Exception: pass
    return None

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}; failed_files = []
    required = ['mice_imputer', 'lgbm_model', 'feature_names']
    missing  = [r for r in required if not os.path.exists(f'{MODEL_DIR}/{r}.pkl')]
    if missing: return None, missing
    for name in sorted(os.listdir(MODEL_DIR)):
        if not name.endswith('.pkl'): continue
        key  = name.replace('.pkl', '')
        path = os.path.join(MODEL_DIR, name)
        obj  = _safe_load_pkl(path)
        if obj is not None: models[key] = obj
        else: failed_files.append(name)
    if failed_files: models['__failed__'] = failed_files
    # AutoGluon — txt файлдан жолды оқып, predictor жүктейді
    ag_txt = os.path.join(MODEL_DIR, 'autogluon_predictor_path.txt')
    if failed_files:
            models['__failed__'] = failed_files

    ag_txt = os.path.join(MODEL_DIR, 'autogluon_predictor_path.txt')
    if os.path.exists(ag_txt):
        try:
            from autogluon.tabular import TabularPredictor
            with open(ag_txt, 'r') as f:
                ag_path = f.read().strip()
            models['autogluon_model'] = TabularPredictor.load(ag_path)
        except Exception as e:
            models['__failed__'] = models.get('__failed__', []) + [f'autogluon ({e})']

    return models, []


# ══════════════════════════════════════════════════════
# БОЛЖАМ
# ══════════════════════════════════════════════════════
def predict(models, raw_input: dict, model_key: str):
    processed = raw_input.copy()
    sbp_raw = processed.pop('SBP_raw', None)
    dbp_raw = processed.pop('DBP_raw', None)
    processed['SBP'] = sbp_to_category(sbp_raw)
    processed['DBP'] = dbp_to_category(dbp_raw)
    df_raw = pd.DataFrame([processed])

    if 'mice_imputer' in models:
        mice = models['mice_imputer']
        mice_cols = list(mice.feature_names_in_) if hasattr(mice, 'feature_names_in_') else models['feature_names']
        df_for_mice = pd.DataFrame(np.nan, index=[0], columns=mice_cols)
        for col in mice_cols:
            if col in df_raw.columns: df_for_mice[col] = df_raw[col].values
            elif col.replace(' ', '_') in df_raw.columns: df_for_mice[col] = df_raw[col.replace(' ', '_')].values
        df_imp = pd.DataFrame(mice.transform(df_for_mice), columns=mice_cols)
    else:
        df_imp = df_raw.copy()

    feat_names = models['feature_names']
    for col in feat_names:
        if col not in df_imp.columns:
            col_space = col.replace('_', ' ')
            if col_space in df_imp.columns: df_imp[col] = df_imp[col_space]
            elif col in df_raw.columns: df_imp[col] = df_raw[col].values
            else: df_imp[col] = 0.0

    X = df_imp[feat_names].copy()
    if 'scaler' in models:
        scaler = models['scaler']
        scaler_cols = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else list(feat_names)
        df_for_scaler = pd.DataFrame(index=[0])
        for c in scaler_cols:
            c_under = c.replace(' ', '_')
            if c in X.columns: df_for_scaler[c] = X[c]
            elif c_under in X.columns: df_for_scaler[c] = X[c_under]
            else: df_for_scaler[c] = 0.0
        scaled_vals = scaler.transform(df_for_scaler)
        for i, c in enumerate(scaler_cols):
            c_under = c.replace(' ', '_')
            if c in X.columns: X[c] = scaled_vals[0, i]
            elif c_under in X.columns: X[c_under] = scaled_vals[0, i]

    model = models[model_key]
    # AutoGluon үшін арнайы болжам
    if model_key == 'autogluon_model':
        from autogluon.tabular import TabularPredictor
        pred = int(model.predict(X)[0])
        prob_series = model.predict_proba(X)
        prob = float(prob_series.iloc[0, 1])
        return pred, prob, X, feat_names
    pred  = int(model.predict(X)[0])
    prob  = float(model.predict_proba(X)[0][1])
    return pred, prob, X, feat_names


# ══════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════
def compute_shap(models, X: pd.DataFrame, model_key: str):
    try:
        explainer_key = f'shap_explainer_{model_key.replace("_model","")}'
        if explainer_key in models:
            explainer = models[explainer_key]   # ← осы жерде қолданылады
        else:
            model = models[model_key]
            if hasattr(model, 'estimators_'): model = model.estimators_[0][1]
            explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list): sv = sv[1]
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)): base = float(base[-1])
        return pd.Series(sv[0], index=X.columns), float(base)
    except Exception:
        return None, None


# ══════════════════════════════════════════════════════
# ГРАФИКТЕР
# ══════════════════════════════════════════════════════
def risk_gauge(prob: float) -> go.Figure:
    color = ("#cc2222" if prob > 0.7 else "#e07000" if prob > 0.4 else "#1a8a4a")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        number={'suffix': '%', 'font': {'size': 44, 'color': color}},
        delta={'reference': 50, 'valueformat': '.1f', 'suffix': '%', 'font': {'size': 16, 'color': '#111128'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#5a5a7a',
                     'tickvals': [0, 25, 50, 75, 100], 'tickfont': {'color': '#111128'}},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': '#f0f2fa', 'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': '#d8f5e8'},
                {'range': [40, 70], 'color': '#fff3cc'},
                {'range': [70, 100], 'color': '#fde0e0'},
            ],
            'threshold': {'line': {'color': '#111128', 'width': 3}, 'thickness': 0.8, 'value': prob * 100}
        },
        title={'text': "CVD Тәуекел Ықтималдығы", 'font': {'size': 15, 'color': '#111128'}}
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor='rgba(0,0,0,0)', font={'color': '#111128'})
    return fig

def shap_waterfall_chart(shap_vals, base_val, prob, n_top=12):
    top    = shap_vals.abs().nlargest(n_top)
    vals   = shap_vals[top.index].sort_values()
    colors = ['#cc2222' if v > 0 else '#2244cc' for v in vals]
    fig = go.Figure(go.Bar(
        x=vals.values, y=[f'<b>{n}</b>' for n in vals.index],
        orientation='h', marker_color=colors,
        text=[f'{v:+.4f}' for v in vals], textposition='outside',
        textfont=dict(size=11, color='#111128'),
        hovertemplate='<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>'
    ))
    fig.add_vline(x=0, line_width=1.5, line_color='#5a5a7a')
    fig.update_layout(
        title=dict(text='SHAP — Белгілердің Болжамға Үлесі', font=dict(size=15, color='#111128')),
        xaxis=dict(title='SHAP мәні', gridcolor='#dde0ee', color='#111128', zeroline=False),
        yaxis=dict(gridcolor='#dde0ee', color='#111128'),
        plot_bgcolor='#f8f9fd', paper_bgcolor='rgba(0,0,0,0)',
        height=420, margin=dict(l=10, r=60, t=50, b=20), font=dict(color='#111128'),
        annotations=[dict(x=0.99, y=0.01, xref='paper', yref='paper',
                          text=f'Базалық: {base_val:.3f} → Соңғы: {prob:.3f}',
                          showarrow=False, font=dict(size=11, color='#5a5a7a'), xanchor='right')]
    )
    return fig

def shap_bar_chart(shap_vals):
    imp    = shap_vals.abs().sort_values(ascending=True).tail(15)
    colors = ['#cc2222' if shap_vals[i] > 0 else '#2244cc' for i in imp.index]
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation='h', marker_color=colors,
        text=[f'{v:.4f}' for v in imp.values], textposition='outside',
        textfont=dict(size=10, color='#111128'),
        hovertemplate='<b>%{y}</b><br>|SHAP|: %{x:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text='Белгілер Маңыздылығы (|SHAP|)', font=dict(size=14, color='#111128')),
        xaxis=dict(title='Орташа |SHAP мәні|', gridcolor='#dde0ee', color='#111128'),
        yaxis=dict(gridcolor='#dde0ee', color='#111128'),
        plot_bgcolor='#f8f9fd', paper_bgcolor='rgba(0,0,0,0)',
        height=420, margin=dict(l=10, r=60, t=50, b=20), font=dict(color='#111128')
    )
    return fig


# ══════════════════════════════════════════════════════
# КЛИНИКАЛЫҚ КЕҢЕСТЕР
# ══════════════════════════════════════════════════════
def get_clinical_advice(prob, sbp, dbp, bmi, glucose, wbc, hgb, spo2, age):
    advices = []

    if sbp > 150 or dbp >= 100:
        level = "қатты жоғарылаған" if (sbp > 170 or dbp > 120) else "жоғарылаған"
        advices.append({
            'type': 'danger', 'icon': '🩺',
            'title': f'Қан қысымы {level}',
            'text': (
                f"Систолалық: <b>{sbp} мм.сын.б.</b>, Диастолалық: <b>{dbp} мм.сын.б.</b>.<br>"
                f"<b>Не істеу керек:</b> 10 минут тыныш отырып, <b>қайта өлшеңіз</b>. "
                f"Нәтиже жоғары болса — дәрігерге жүгінуіңізді ұсынамыз. "
                f"Давление түсіретін дәрі қабылдасаңыз, дозасы мен уақытын тексеріңіз. "
                f"Тұзды тағамнан, кофеден аулақ болыңыз."
            )
        })

    if sbp > 140 or dbp >= 90:
        advices.append({
            'type': 'info', 'icon': '🍵',
            'title': 'Фитотерапия — қосымша қолдау',
            'text': (
                "Дәрімен қатар дәрігер рұқсатымен қосымша ем ретінде: "
                "<b>ибергин шайы</b>, <b>валериана шайы</b>, <b>жүзім жапырақ шайы</b> "
                "тыныштандырушы және гипотензивті әсер береді. "
                "Көк шай мен кофені азайтыңыз — олар қысымды арттырады."
            )
        })

    if glucose > 7.0:
        advices.append({
            'type': 'danger', 'icon': '🍬',
            'title': 'Қан глюкозасы жоғары',
            'text': (
                f"Глюкоза: <b>{glucose:.1f} ммоль/л</b> (норма: 3.9–7.0).<br>"
                f"<b>Не істеу керек:</b> Тәтті, ақ нан, картоп тұтынуды азайтыңыз. "
                f"Қант диабеті бар болса — дәрігермен кеңесіп, дозаны реттеңіз."
            )
        })

    if bmi > 30:
        advices.append({
            'type': 'danger', 'icon': '⚖️',
            'title': f'Семіздік (BMI = {bmi:.1f} кг/м²)',
            'text': (
                "Артық салмақ жүрекке қосымша жүктеме жасайды.<br>"
                "<b>Не істеу керек:</b> Тәулігіне 30 мин жаяу жүру, "
                "майлы және қантты тағамды азайту, диетологпен кеңесу."
            )
        })

    if spo2 < 95:
        advices.append({
            'type': 'danger', 'icon': '🫁',
            'title': f'Оттегі қанықтылығы төмен (SpO₂ = {spo2:.1f}%)',
            'text': (
                "Норма: ≥ 95%.<br>"
                "<b>Не істеу керек:</b> Таза ауаға шығыңыз, терең тыныс алыңыз. "
                "Тыныс алу қиын болса — дереу дәрігерге жүгініңіз."
            )
        })

    if hgb < 110:
        advices.append({
            'type': 'danger', 'icon': '🩸',
            'title': f'Гемоглобин төмен — анемия (HGB = {hgb:.0f} г/л)',
            'text': (
                "Норма: ер — 130+ г/л, әйел — 120+ г/л.<br>"
                "<b>Не істеу керек:</b> Темір бар тағамдарды (ет, бауыр) жеңіз. "
                "Дәрігер тағайындаса — темір препараттарын қабылдаңыз."
            )
        })

    if wbc > 10:
        advices.append({
            'type': 'danger', 'icon': '🦠',
            'title': f'Лейкоциттер жоғары — қабыну белгісі (WBC = {wbc:.1f})',
            'text': (
                "Норма: 4–10 ×10⁹/л.<br>"
                "<b>Не істеу керек:</b> Ауруханада толық қан анализін тапсырыңыз — "
                "жасырын инфекция немесе қабыну болуы мүмкін."
            )
        })

    if age > 60 and prob > 0.5:
        advices.append({
            'type': 'info', 'icon': '👴',
            'title': '60+ жас — профилактикалық тексеру маңызды',
            'text': (
                "Жасқа байланысты жүрек ауруы қаупі артады.<br>"
                "<b>Ұсыным:</b> Жылына 1 рет ЭКГ, эхокардиография, "
                "жалпы қан анализін тапсырып тұрыңыз."
            )
        })

    if prob < 0.3 and not advices:
        advices.append({
            'type': 'success', 'icon': '🌿',
            'title': 'Тамаша нәтиже! Профилактиканы жалғастырыңыз',
            'text': (
                "CVD тәуекелі төмен.<br>"
                "<b>Сақтау үшін:</b> Жылына бір медициналық тексеруден өтіңіз, "
                "физикалық белсенділікті сақтаңыз, темекіден аулақ болыңыз."
            )
        })

    return advices


# ══════════════════════════════════════════════════════
# МОДЕЛЬ МӘЛІМЕТТЕРІ
# ══════════════════════════════════════════════════════
MODEL_LABELS = {
    'lgbm_model':     ' LightGBM (Tuned)',
    'xgb_model':      ' XGBoost (Tuned)',
    'cat_model':      ' CatBoost (Tuned)',
    'rf_model':       ' Random Forest (Tuned)',
    'dt_model':       ' Decision Tree (Tuned)',
    'stacking_model': ' Stacking Ensemble',
    'autogluon_model': ' AutoGluon',
}
MODEL_INFO = {
    'lgbm_model':     ('Recall: 94.2%', 'ROC AUC: 96.8%', 'SHAP қолдайды'),
    'xgb_model':      ('Recall: 92.9%', 'ROC AUC: 97.0%', 'SHAP қолдайды'),
    'cat_model':      ('Recall: 91.7%', 'ROC AUC: 95.5%', 'SHAP қолдайды'),
    'rf_model':       ('Recall: 87.5%', 'ROC AUC: 94.9%', 'SHAP қолдайды'),
    'dt_model':       ('Recall: 87.5%', 'ROC AUC: 86.9%', 'SHAP қолдайды'),
    'stacking_model': ('Recall: 90.4%', 'ROC AUC: 96.8%', 'SHAP жартылай'),
    'autogluon_model': ('Recall: —', 'ROC AUC: —', 'SHAP қолдайды')
}

models, missing = load_models()
available_models = {}
model_choice = 'lgbm_model'
threshold    = 0.50
show_shap    = True

if models is not None:
    available_models = {k: v for k, v in MODEL_LABELS.items()
                    if k in models and not k.startswith('__')}

# ══════════════════════════════════════════════════════
# SIDEBAR (тек форма/нәтиже беттерінде)
# ══════════════════════════════════════════════════════
if st.session_state.page != 'warning':
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:20px 0 10px;'>
            <div style='font-size:48px;'>🫀</div>
            <div style='font-family:"DM Serif Display",serif; font-size:22px;
                        color:#f0f4ff !important; margin-top:8px; font-weight:700;'>
                CVD · Тәуекел Анализі
            </div>
            <div style='font-size:11px; color:#8fa0cc !important; margin-top:4px;
                        letter-spacing:2px; text-transform:uppercase;'>
                Машиналық Оқыту · XAI
            </div>
        </div>
        <hr style='border-color:#2a2d50; margin:16px 0;'>
        """, unsafe_allow_html=True)

        st.markdown("""<p style='color:#c8d4f5 !important; font-weight:700;
            font-size:15px; margin-bottom:6px;'>🤖 Модель Таңдау</p>""",
            unsafe_allow_html=True)

        if available_models:
            model_choice = st.selectbox("Модель:", list(available_models.keys()),
                format_func=lambda k: available_models[k], label_visibility='collapsed')
            if model_choice in MODEL_INFO:
                r, auc, shap_note = MODEL_INFO[model_choice]
                st.markdown(f"""
                <div style='background:#252845; border:1px solid #3a3f70; border-radius:10px;
                            padding:12px 14px; margin-top:8px;'>
                    <div style='font-size:13px; margin-bottom:3px;'>
                        <span style='color:#74aaff !important;'>📊</span>
                        <span style='color:#dde3f5 !important; font-weight:600;'> {r}</span>
                    </div>
                    <div style='font-size:13px; margin-bottom:3px;'>
                        <span style='color:#74aaff !important;'>📈</span>
                        <span style='color:#dde3f5 !important; font-weight:600;'> {auc}</span>
                    </div>
                    <div style='font-size:13px;'>
                        <span style='color:#74aaff !important;'>🔍</span>
                        <span style='color:#b8c8e8 !important;'> {shap_note}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#2a2d50; margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("""<p style='color:#c8d4f5 !important; font-weight:700;
            font-size:15px; margin-bottom:6px;'>⚙️ Болжам Баптаулары</p>""",
            unsafe_allow_html=True)

        threshold = st.slider("Шекті мән (Threshold)", 0.30, 0.80, 0.50, 0.01,
                              help="Ауру деп санау ықтималдығы осы мәннен жоғары болса")
        show_shap = st.toggle("SHAP Визуализация", value=True)

        st.markdown("<hr style='border-color:#2a2d50; margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:12px; line-height:2.0; background:#1e2240;
                    border:1px solid #2a2d50; border-radius:10px; padding:12px 14px;'>
            <span style='color:#ffd060 !important; font-weight:700;'>📊 SBP Категориялар</span><br>
            <span style='color:#c8d4f5 !important;'>0 = Норма (≤150)</span><br>
            <span style='color:#c8d4f5 !important;'>1 = Жоғарылау (151–170)</span><br>
            <span style='color:#c8d4f5 !important;'>2 = Гипертония (&gt;170)</span><br>
            <span style='color:#ffd060 !important; font-weight:700;'>📊 DBP Категориялар</span><br>
            <span style='color:#c8d4f5 !important;'>0 = Норма (&lt;100)</span><br>
            <span style='color:#c8d4f5 !important;'>1 = Жоғарылау (100–120)</span><br>
            <span style='color:#c8d4f5 !important;'>2 = Гипертония (&gt;120)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#2a2d50; margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:12px; line-height:2.0; background:#1e2240;
                    border:1px solid #2a2d50; border-radius:10px; padding:12px 14px;'>
            <span style='color:#ff7070 !important; font-weight:700;'>🔴 Қызыл SHAP</span>
            <span style='color:#c8d4f5 !important;'> → Тәуекелді арттырады</span><br>
            <span style='color:#74aaff !important; font-weight:700;'>🔵 Көк SHAP</span>
            <span style='color:#c8d4f5 !important;'> → Тәуекелді азайтады</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#2a2d50; margin:16px 0;'>", unsafe_allow_html=True)
        if st.button("⬅️  Ескертуге қайту", key="back_to_warning"):
            st.session_state.page = 'warning'
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
#  БЕТ 1: ЕСКЕРТУ
# ══════════════════════════════════════════════════════════════════════════
def page_warning():
    st.markdown("""
    <div style='text-align:center; padding:40px 0 20px;'>
        <div style='font-size:72px; margin-bottom:12px;'>🫀</div>
        <h1 style='font-size:42px; margin:0; font-weight:700; color:#111128 !important;'>
            CVD Тәуекел Болжаушы
        </h1>
        <p style='color:#3a3a5c !important; font-size:16px; margin-top:10px;
                  max-width:680px; margin-left:auto; margin-right:auto;'>
            Жүрек-Қан Тамыр Ауруын Machine Learning арқылы анықтаудан бұрын<br>
            келесі ескертулерді мұқият оқыңыз
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#d0d4e8; margin-bottom:32px;'>", unsafe_allow_html=True)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.markdown("""
        <h2 style='color:#111128 !important; font-weight:700; margin-bottom:20px;'>
            ⚠️ Өлшеу алдындағы маңызды ескертулер
        </h2>
        """, unsafe_allow_html=True)

        items = [
            ("🕐", "Тынышталған кезде өлшеңіз",
             "Жүгіріп келген, жаттығудан кейін немесе физикалық жүктемеден кейін "
             "дереу өлшемеңіз. Кем дегенде <b>5–10 минут</b> отырып тынышталыңыз — "
             "қан қысымы мен жүрек соғу жиілігі нақты болады."),

            ("☕", "Кофе, шай, шоколад ішпеңіз",
             "Өлшеуден кемінде <b>1–2 сағат бұрын</b> кофе, қара шай, жасыл шай, "
             "энергетикалық сусындар немесе шоколад жемеңіз. "
             "Кофеин қан қысымын уақытша арттырып, нәтижені бұрмалайды."),

            ("🚬", "Темекі шекпеңіз",
             "Өлшеуден <b>30 минут бұрын</b> темекі шекпеңіз. "
             "Никотин тамырларды тарылтып, систолалық қысымды 10–20 мм.сын.б. дейін арттырады."),

            ("💊", "Давление дәрісін тұрақты қабылдаңыз",
             "Гипертония дәрісін <b>бір уақытта, бір дозамен</b> ішіңіз. "
             "Дәрі ішпей қалған күні өлшеу нәтижесі жоғары болуы мүмкін — "
             "давлениеңіздің тұрақтылығын бақылаңыз."),

            ("🍽️", "Ас ішкеннен кейін 30–60 мин күтіңіз",
             "Тамақ ішкен соң дереу өлшемеңіз. Асқорыту кезінде жүрек жиірек соғады. "
             "<b>30–60 минут</b> өткен соң өлшеңіз."),

            ("🧘", "Стресс пен қобалжу кезінде өлшемеңіз",
             "Дауласқан, қорқыныш немесе қатты қобалжыған кезде өлшемеңіз. "
             "Эмоционалды стресс адреналин бөліп, қан қысымын уақытша <b>15–30 мм</b> "
             "дейін жоғарылата алады."),

            ("🌡️", "Суық / ыстық ортадан кірген соң күтіңіз",
             "Суықта немесе ыстықта болып кірген соң кем дегенде <b>5–10 минут</b> "
             "бөлме температурасына үйренсін — температура айырмашылығы тамыр тонусына әсер етеді."),

            ("📐", "Дұрыс позада өлшеңіз",
             "Отырып өлшеңіз: арқаңызды тіктеп, аяқтарыңыз еденде, "
             "қолыңыз жүрек деңгейінде тұрсын."),
        ]

        for icon, title, text in items:
            st.markdown(f"""
            <div class='warning-card'>
                <div class='warning-icon'>{icon}</div>
                <div>
                    <div class='warning-title'>{title}</div>
                    <div class='warning-text'>{text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_side:
        st.markdown("""
        <div style='background:linear-gradient(160deg,#d2d2d4,#d2d2d4);
                    border-radius:20px; padding:28px 24px; position:sticky; top:20px;'>
            <div style='font-size:36px; text-align:center; margin-bottom:12px;'>📋</div>
            <h3 style='color:#f0f4ff !important; text-align:center; margin:0 0 16px; font-size:18px;'>
                Болжам алдындағы тізім
            </h3>
            <div style='font-size:14px; line-height:2.4;'>
                <div style='color:#80d8a0 !important;'>✅ &nbsp;5–10 мин тынышталдым</div>
                <div style='color:#80d8a0 !important;'>✅ &nbsp;Кофе/шай ішпедім</div>
                <div style='color:#80d8a0 !important;'>✅ &nbsp;Темекі шекпедім</div>
                <div style='color:#80d8a0 !important;'>✅ &nbsp;Дұрыс отырдым</div>
                <div style='color:#80d8a0 !important;'>✅ &nbsp;Тыныш күйдемін</div>
                <div style='color:#80d8a0 !important;'>✅ &nbsp;Давление дәрісім тұрақты</div>
                <div style='color:#80d8a0 !important;'>✅ &nbsp;Тамақтан 30+ мин өтті</div>
            </div>
            <hr style='border-color:#3a3f70; margin:20px 0;'>
            <div style='font-size:11px; color:#8fa0cc !important; text-align:center; line-height:1.7;'>
                ⚕️ Бұл жүйе тек ақпараттық мақсатта жасалған.<br>
                Нәтиже медициналық диагноз болып табылмайды.<br>
                Дәрігер кеңесін алуды ұмытпаңыз.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#d0d4e8;'>", unsafe_allow_html=True)

    _, col_c, _ = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <p style='text-align:center; color:#3a3a5c !important; font-size:14px; margin-bottom:10px;'>
            Барлық шарттарды орындадыңыз ба? Онда болжамды бастауға дайынсыз!
        </p>
        """, unsafe_allow_html=True)
        st.markdown('<div class="btn-green">', unsafe_allow_html=True)
        if st.button("✅  Дайынмын — Болжамға өту →", key="go_to_form_btn",
                     use_container_width=True):
            st.session_state.page = 'form'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <hr style='border-color:#d0d4e8; margin-top:40px;'>
    <div style='text-align:center; padding:20px; font-size:12px; color:#5a5a7a !important;'>
        🫀 CVD Тәуекел Болжаушы · Дипломдық жұмыс · ҚазҰУ · 2026<br>
        Әл-Фараби атындағы Қазақ Ұлттық Университеті
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  БЕТ 2: ФОРМА
# ══════════════════════════════════════════════════════════════════════════
def page_form(models, available_models, model_choice, threshold):
    if models is None:
        st.error(f"Модельдер табылмады! Жетіспейтін: `{', '.join(missing)}`")
        st.stop()

    st.markdown("""
    <div style='text-align:center; padding:30px 0 10px;'>
        <h1 style='font-size:40px; margin:0; color:#111128 !important; font-weight:700;'>
            Жүрек-Қан Тамыр Тәуекелін Болжау
        </h1>
        <p style='color:#3a3a5c !important; font-size:16px; margin-top:8px;'>
            Клиникалық биомаркерлерді енгізіңіз — модель CVD тәуекелін нақты есептейді
        </p>
    </div>
    <hr style='border-color:#d0d4e8; margin-bottom:30px;'>
    """, unsafe_allow_html=True)

    with st.form("patient_form"):
        st.markdown("""
        <h3 style='color:#111128 !important; font-family:"DM Serif Display",serif; margin-bottom:16px;'>
            📋 Пациент Деректерін Енгізу
        </h3>""", unsafe_allow_html=True)

        st.markdown("""<div style='font-size:12px; font-weight:700; letter-spacing:2px;
            text-transform:uppercase; color:#5a5a7a !important; margin-bottom:8px;'>
            🧑 ДЕМОГРАФИЯЛЫҚ КӨРСЕТКІШТЕР</div>""", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: age    = st.number_input("Жасы (Age)", 1, 120, 45)
        with c2: gender = st.selectbox("Жынысы", options=[0, 1],
                                       format_func=lambda x: "Әйел (0)" if x == 0 else "Еркек (1)")
        with c3: weight = st.number_input("Салмағы (Weight)", 20.0, 200.0, 75.0, 0.5)
        with c4: height = st.number_input("Бойы (Height)", 50.0, 250.0, 170.0, 1.0)
        with c5: bmi    = st.number_input("ДСИ (BMI, кг/м²)", 10.0, 60.0, 25.0, 0.1)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style='font-size:12px; font-weight:700; letter-spacing:2px;
            text-transform:uppercase; color:#5a5a7a !important; margin-bottom:8px;'>
            💓 ҚАН ҚЫСЫМЫ ЖӘНЕ ЖҮРЕК</div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: sbp = st.number_input("Систолалық қысым (SBP, мм.сын.б.)", 60, 250, 120)
        with c2: dbp = st.number_input("Диастолалық қысым (DBP, мм.сын.б.)", 40, 150, 80)
        with c3: hr  = st.number_input("ЖСЖ (HR, уд/мин)", 30, 200, 72)

        sbp_cat = sbp_to_category(sbp); dbp_cat = dbp_to_category(dbp)
        SBP_LABELS = {0:"0 — Норма (≤150)", 1:"1 — Жоғарылау (151–170)", 2:"2 — Гипертония (>170)"}
        DBP_LABELS = {0:"0 — Норма (<100)", 1:"1 — Жоғарылау (100–120)", 2:"2 — Гипертония (>120)"}
        SBP_COLORS = DBP_COLORS = {0:"#1a8a4a", 1:"#e07000", 2:"#cc2222"}

        c1, c2 = st.columns(2)
        with c1: spo2 = st.number_input("Оттегі қанықтылығы (SpO₂, %)", 70.0, 100.0, 97.0, 0.1)
        with c2:
            pp = sbp - dbp; map_val = dbp + pp / 3
            st.markdown(f"""
            <div style='background:#eef2ff; border:1px solid #b0bcee; border-radius:10px;
                        padding:14px 18px; margin-top:4px;'>
                <div style='color:#5a5a7a !important; font-size:11px; font-weight:700;
                            text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>
                    Есептелген Көрсеткіштер
                </div>
                <div style='font-size:13px; font-weight:600; margin-bottom:4px;'>
                    <span style='color:#111128 !important;'>💓 Пульс қысымы: </span>
                    <span style='color:#c05000 !important; font-weight:700;'>{pp} мм</span>
                    <span style='color:#5a5a7a !important; font-size:11px;'> (норма: 30–50)</span>
                </div>
                <div style='font-size:13px; font-weight:600; margin-bottom:6px;'>
                    <span style='color:#111128 !important;'>🩺 ОАҚ (MAP): </span>
                    <span style='color:#1a4abf !important; font-weight:700;'>{map_val:.0f} мм</span>
                    <span style='color:#5a5a7a !important; font-size:11px;'> (норма: 70–100)</span>
                </div>
                <hr style='border-color:#c8d0ee; margin:8px 0;'>
                <div style='font-size:12px; font-weight:600; margin-bottom:3px;'>
                    <span style='color:#5a5a7a !important;'>📊 SBP: </span>
                    <span style='color:{SBP_COLORS[sbp_cat]} !important; font-weight:700;'>
                        {SBP_LABELS[sbp_cat]}</span>
                </div>
                <div style='font-size:12px; font-weight:600;'>
                    <span style='color:#5a5a7a !important;'>📊 DBP: </span>
                    <span style='color:{DBP_COLORS[dbp_cat]} !important; font-weight:700;'>
                        {DBP_LABELS[dbp_cat]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style='font-size:12px; font-weight:700; letter-spacing:2px;
            text-transform:uppercase; color:#5a5a7a !important; margin-bottom:8px;'>
            🩸 ҚАН АНАЛИЗІ</div>""", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: plt_val = st.number_input("Тромбоциттер (PLT, ×10⁹/л)", 50.0, 800.0, 250.0, 1.0)
        with c2: hgb     = st.number_input("Гемоглобин (HGB, г/л)", 60.0, 220.0, 130.0, 0.5)
        with c3: wbc     = st.number_input("Лейкоциттер (WBC, ×10⁹/л)", 1.0, 30.0, 6.5, 0.1)
        with c4: rbc     = st.number_input("Эритроциттер (RBC, ×10¹²/л)", 2.0, 8.0, 4.5, 0.01)
        c1, c2, c3 = st.columns(3)
        with c1: hct        = st.number_input("Гематокрит (HCT, %)", 15.0, 65.0, 40.0, 0.1)
        with c2: creatinine = st.number_input("Креатинин (мкмоль/л)", 30.0, 1000.0, 80.0, 0.5)
        with c3: glucose    = st.number_input("Глюкоза (ммоль/л)", 2.0, 30.0, 5.5, 0.1)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style='font-size:12px; font-weight:700; letter-spacing:2px;
            text-transform:uppercase; color:#5a5a7a !important; margin-bottom:8px;'>
            🫀 БАУЫР ФЕРМЕНТТЕРІ</div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: alt       = st.number_input("АЛТ (Ед/л)", 1.0, 500.0, 25.0, 0.5)
        with c2: ast       = st.number_input("АСТ (Ед/л)", 1.0, 500.0, 22.0, 0.5)
        with c3: bilirubin = st.number_input("Жалпы Билирубин (мкмоль/л)", 1.0, 300.0, 12.0, 0.5)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍  CVD Тәуекелін Анықтау")

    if submitted:
        raw_input = {
            'Age': age, 'Gender': gender, 'Weight': weight, 'Height': height, 'BMI': bmi,
            'SBP_raw': sbp, 'DBP_raw': dbp, 'Heart_Rate': hr, 'Oxygen_Saturation': spo2,
            'Platelets_(PLT)': plt_val, 'Hemoglobin_(HGB)': hgb, 'Leukocytes_(WBC)': wbc,
            'Erythrocytes_(RBC)': rbc, 'Hematocrit_(HCT)': hct, 'Creatinine': creatinine,
            'ALT': alt, 'AST': ast, 'Total_Bilirubin': bilirubin, 'Glucose': glucose,
        }
        with st.spinner('⏳ Модель болжауды есептеуде...'):
            pred, prob, X_input, feat_names = predict(models, raw_input, model_choice)
        shap_vals_res, base_val_res = None, None
        if show_shap:
            shap_vals_res, base_val_res = compute_shap(models, X_input, model_choice)

        st.session_state.result_data = {
            'raw_input': raw_input, 'pred': pred, 'prob': prob,
            'X_input': X_input, 'feat_names': feat_names,
            'shap_vals': shap_vals_res, 'base_val': base_val_res,
            'model_choice': model_choice, 'threshold': threshold, 'show_shap': show_shap,
            'sbp': sbp, 'dbp': dbp, 'bmi': bmi, 'hr': hr, 'spo2': spo2,
            'age': age, 'gender': gender, 'glucose': glucose,
            'hgb': hgb, 'wbc': wbc, 'creatinine': creatinine,
        }
        st.session_state.page = 'result'
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
#  БЕТ 3: НӘТИЖЕ
# ══════════════════════════════════════════════════════════════════════════
def page_result(models, available_models):
    data = st.session_state.result_data
    if data is None:
        st.warning("Деректер жоқ.")
        if st.button("⬅️ Формаға қайту"):
            st.session_state.page = 'form'; st.rerun()
        return

    prob         = data['prob'];         pred         = data['pred']
    X_input      = data['X_input'];      model_choice = data['model_choice']
    threshold    = data['threshold'];    show_shap    = data['show_shap']
    shap_vals    = data['shap_vals'];    base_val     = data['base_val']
    raw_input    = data['raw_input']
    sbp          = data['sbp'];          dbp          = data['dbp']
    bmi          = data['bmi'];          spo2         = data['spo2']
    age          = data['age'];          gender       = data['gender']
    glucose      = data['glucose'];      hgb          = data['hgb']
    wbc          = data['wbc']

    label    = 1 if prob >= threshold else 0
    risk_pct = prob * 100

    st.markdown("""
    <div style='text-align:center; padding:20px 0 6px;'>
        <h1 style='font-size:38px; margin:0; font-weight:700; color:#111128 !important;'>
            📊 Болжам Нәтижесі
        </h1>
    </div>
    <hr style='border-color:#d0d4e8; margin-bottom:24px;'>
    """, unsafe_allow_html=True)

    scaler_used = 'scaler' in models
    st.markdown(f"""
    <div style='display:inline-block; background:{"#eaffea" if scaler_used else "#fffbe6"};
                border:1px solid {"#6acc6a" if scaler_used else "#e0c060"};
                border-radius:20px; padding:4px 14px; font-size:12px; margin-bottom:16px;'>
        {"✅ StandardScaler қолданылды" if scaler_used else "⚠️ Scaler қолданылмады"}
        &nbsp;·&nbsp;
        <span style='color:#5a5a7a !important;'>
            Модель: {available_models.get(model_choice, model_choice)}</span>
        &nbsp;·&nbsp;
        <span style='color:#5a5a7a !important;'>
            SBP кат: <b>{sbp_to_category(sbp)}</b> · DBP кат: <b>{dbp_to_category(dbp)}</b>
        </span>
    </div>
    """, unsafe_allow_html=True)

    col_res, col_gauge = st.columns([1, 1])
    with col_res:
        if label == 1:
            st.markdown(f"""
            <div class='result-sick'>
                <div style='font-size:56px; margin-bottom:10px;'>🚨</div>
                <div style='font-family:"DM Serif Display",serif; font-size:32px;
                            color:#aa1111 !important; font-weight:700;'>Жоғары Тәуекел</div>
                <div style='font-size:18px; color:#cc2222 !important; margin-top:8px; font-weight:700;'>
                    CVD ықтималдығы: {risk_pct:.1f}%</div>
                <div style='font-size:13px; color:#883333 !important; margin-top:12px;'>
                    {available_models.get(model_choice, model_choice)} · Threshold: {threshold:.2f}</div>
                <div style='font-size:12px; color:#662222 !important; margin-top:16px;
                            padding:10px; background:rgba(200,50,50,0.08);
                            border-radius:10px; border:1px solid rgba(200,50,50,0.2);'>
                    ⚠️ Бұл ақпарат тек ақпараттық мақсатта берілген. Дәрігерге жүгінуді ұсынамыз.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-healthy'>
                <div style='font-size:56px; margin-bottom:10px;'>✅</div>
                <div style='font-family:"DM Serif Display",serif; font-size:32px;
                            color:#116622 !important; font-weight:700;'>Төмен Тәуекел</div>
                <div style='font-size:18px; color:#1a8833 !important; margin-top:8px; font-weight:700;'>
                    CVD ықтималдығы: {risk_pct:.1f}%</div>
                <div style='font-size:13px; color:#336644 !important; margin-top:12px;'>
                    {available_models.get(model_choice, model_choice)} · Threshold: {threshold:.2f}</div>
                <div style='font-size:12px; color:#224433 !important; margin-top:16px;
                            padding:10px; background:rgba(50,180,100,0.08);
                            border-radius:10px; border:1px solid rgba(50,180,100,0.2);'>
                    ✔️ Профилактикалық тексерулерді жалғастыруды ұсынамыз.
                </div>
            </div>""", unsafe_allow_html=True)

    with col_gauge:
        st.plotly_chart(risk_gauge(prob), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    prob_color = "#cc2222" if prob > 0.5 else "#1a8a4a"
    with m1:
        st.markdown(f"""<div class='card'><div class='card-header'>Ауру Ықтималдығы</div>
        <div class='card-value' style='color:{prob_color} !important;'>{prob:.1%}</div></div>""",
        unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='card'><div class='card-header'>Сау Ықтималдығы</div>
        <div class='card-value' style='color:#1a4abf !important;'>{1 - prob:.1%}</div></div>""",
        unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='card'><div class='card-header'>Пульс Қысымы</div>
        <div class='card-value' style='color:#c06000 !important;'>{sbp - dbp}
        <span style='font-size:16px;'>мм</span></div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class='card'><div class='card-header'>ОАҚ (MAP)</div>
        <div class='card-value' style='color:#7722aa !important;'>{dbp + (sbp - dbp) / 3:.0f}
        <span style='font-size:16px;'>мм</span></div></div>""", unsafe_allow_html=True)

    # ── КЛИНИКАЛЫҚ КЕҢЕСТЕР ──
    st.markdown("<hr style='border-color:#d0d4e8;'>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='color:#111128 !important; font-family:"DM Serif Display",serif; font-weight:700;'>
        💊 Пациентке Клиникалық Кеңестер
    </h2>
    <p style='color:#3a3a5c !important; font-size:14px; margin-bottom:20px;'>
        Нәтижеге сүйене отырып, жеке параметрлеріңіз бойынша ұсыныстар:
    </p>
    """, unsafe_allow_html=True)

    advices = get_clinical_advice(prob, sbp, dbp, bmi, glucose, wbc, hgb, spo2, age)
    bg_map     = {'danger': '#fff5f5', 'info': '#f0f4ff', 'success': '#f0fff6'}
    border_map = {'danger': '#f0a0a0', 'info': '#a0b4ee', 'success': '#80c8a0'}
    title_map  = {'danger': '#aa1111', 'info': '#1a3aaa', 'success': '#116622'}

    if advices:
        for adv in advices:
            bg = bg_map.get(adv['type'], '#f8f8f8')
            bc = border_map.get(adv['type'], '#ccc')
            tc = title_map.get(adv['type'], '#111128')
            st.markdown(f"""
            <div style='background:{bg}; border:1.5px solid {bc}; border-radius:14px;
                        padding:16px 20px; margin-bottom:12px;'>
                <div style='font-size:16px; font-weight:700; color:{tc} !important; margin-bottom:6px;'>
                    {adv['icon']} &nbsp;{adv['title']}
                </div>
                <div style='font-size:13px; color:#3a3a5c !important; line-height:1.7;'>
                    {adv['text']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Барлық параметрлер қалыпты деңгейде. Профилактиканы жалғастырыңыз!")

    # ── МОДЕЛЬДЕР САЛЫСТЫРУЫ ──
    st.markdown("<hr style='border-color:#d0d4e8;'>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='color:#111128 !important; font-family:"DM Serif Display",serif; font-weight:700;'>
        🔄 Барлық Модельдердің Болжамы
    </h2>""", unsafe_allow_html=True)
    st.caption("Бір пациент үшін барлық сақталған модельдердің CVD ықтималдығы")

    comparison_rows = []
    for mkey, mlabel in available_models.items():
        try:
            _, mprob, _, _ = predict(models, raw_input, mkey)
            mlabel_clean = 1 if mprob >= threshold else 0
            comparison_rows.append({'Модель': mlabel, 'Ықтималдық': round(mprob, 4),
                'Болжам': '🔴 Ауру' if mlabel_clean == 1 else '🟢 Сау',
                'Тәуекел (%)': round(mprob * 100, 1)})
        except Exception: pass

    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows).sort_values('Ықтималдық', ascending=False)
        fig_comp = go.Figure(go.Bar(
            x=comp_df['Тәуекел (%)'], y=comp_df['Модель'], orientation='h',
            marker_color=['#cc2222' if p >= threshold*100 else '#1a8a4a' for p in comp_df['Тәуекел (%)']],
            text=[f"{p}%" for p in comp_df['Тәуекел (%)']],
            textposition='outside', textfont=dict(size=12, color='#111128'),
        ))
        fig_comp.add_vline(x=threshold*100, line_width=2, line_dash='dash',
                           line_color='#c06000', annotation_text=f'Threshold: {threshold:.0%}',
                           annotation_font_color='#c06000')
        fig_comp.update_layout(
            plot_bgcolor='#f8f9fd', paper_bgcolor='rgba(0,0,0,0)',
            height=320, margin=dict(l=10, r=80, t=20, b=20),
            xaxis=dict(range=[0, 110], title='CVD Ықтималдығы (%)', gridcolor='#dde0ee', color='#111128'),
            yaxis=dict(gridcolor='#dde0ee', color='#111128'), font=dict(color='#111128')
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.dataframe(comp_df.reset_index(drop=True), use_container_width=True)

    # ── SHAP ──
    if show_shap:
        st.markdown("<hr style='border-color:#d0d4e8;'>", unsafe_allow_html=True)
        st.markdown("""
        <h2 style='color:#111128 !important; font-family:"DM Serif Display",serif; font-weight:700;'>
            🧠 Explainable AI — SHAP Талдауы
        </h2>""", unsafe_allow_html=True)
        st.info(
            "**SHAP** (SHapley Additive exPlanations) — ойын теориясына негізделген XAI әдісі. "
            "🔴 Қызыл → Тәуекелді **арттырады** · 🔵 Көк → **Азайтады**"
        )

        if shap_vals is not None:
            col_wf, col_bar = st.columns(2)
            with col_wf:
                st.plotly_chart(shap_waterfall_chart(shap_vals, base_val, prob, n_top=12),
                                use_container_width=True)
            with col_bar:
                st.plotly_chart(shap_bar_chart(shap_vals), use_container_width=True)

            st.markdown("""<h3 style='color:#111128 !important; font-weight:700;'>
                📋 Белгілердің Толық SHAP Кестесі</h3>""", unsafe_allow_html=True)

            shap_table = pd.DataFrame({
                'Белгі': shap_vals.index, 'SHAP мәні': shap_vals.values.round(5),
                '|SHAP|': shap_vals.abs().values.round(5),
                'Бағыты': ['↑ Арттырады' if v > 0 else '↓ Азайтады' for v in shap_vals],
                'Маңыздылығы': ['⭐⭐⭐' if abs(v) > 0.1 else '⭐⭐' if abs(v) > 0.04 else '⭐'
                                for v in shap_vals]
            }).sort_values('|SHAP|', ascending=False).reset_index(drop=True)
            shap_table.index += 1

            def color_dir(val): return ('color:#cc2222; font-weight:600' if '↑' in val
                                        else 'color:#2244cc; font-weight:600')
            def color_shap(val):
                if val > 0.05: return 'color:#cc2222'
                if val < -0.05: return 'color:#2244cc'
                return 'color:#111128'

            styled = (shap_table.style
                      .applymap(color_dir, subset=['Бағыты'])
                      .applymap(color_shap, subset=['SHAP мәні'])
                      .background_gradient(subset=['|SHAP|'], cmap='YlOrRd', vmin=0))
            st.dataframe(styled, use_container_width=True, height=400)

            st.markdown("""<h3 style='color:#111128 !important; font-weight:700;'>
                💊 Клиникалық Интерпретация (SHAP бойынша)</h3>""", unsafe_allow_html=True)

            top3_risk = shap_vals[shap_vals > 0].nlargest(3)
            top3_prot = shap_vals[shap_vals < 0].nsmallest(3)
            icol1, icol2 = st.columns(2)
            with icol1:
                st.markdown("""<h4 style='color:#cc2222 !important; font-weight:700;'>
                    🔴 Тәуекелді Арттыратын Факторлар</h4>""", unsafe_allow_html=True)
                if len(top3_risk) > 0:
                    for feat, val in top3_risk.items():
                        st.markdown(f"""
                        <div style='background:#fff0f0; border:1px solid #e0a0a0;
                                    border-radius:10px; padding:12px 16px; margin:6px 0;'>
                            <b style='color:#aa1111 !important;'>{feat}</b>
                            <span style='float:right; color:#cc2222 !important; font-weight:700;'>
                                +{val:.4f}</span><br>
                            <span style='color:#883333 !important; font-size:12px;'>
                                CVD тәуекелін {abs(val)*100:.1f} п.б. арттырады</span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.success("Тәуекелді арттыратын фактор анықталмады")
            with icol2:
                st.markdown("""<h4 style='color:#1a4abf !important; font-weight:700;'>
                    🔵 Қорғаушы Факторлар</h4>""", unsafe_allow_html=True)
                if len(top3_prot) > 0:
                    for feat, val in top3_prot.items():
                        st.markdown(f"""
                        <div style='background:#f0f4ff; border:1px solid #a0b4ee;
                                    border-radius:10px; padding:12px 16px; margin:6px 0;'>
                            <b style='color:#1a3aaa !important;'>{feat}</b>
                            <span style='float:right; color:#2244cc !important; font-weight:700;'>
                                {val:.4f}</span><br>
                            <span style='color:#334488 !important; font-size:12px;'>
                                CVD тәуекелін {abs(val)*100:.1f} п.б. азайтады</span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("Анық қорғаушы фактор анықталмады")
        else:
            st.warning("⚠️ SHAP есептеу сәтсіз болды. LightGBM немесе XGBoost таңдаңыз.")

    # ── CSV + ҚАЙТА ӨЛШЕУ ──
    st.markdown("<hr style='border-color:#d0d4e8;'>", unsafe_allow_html=True)
    result_dict = {
        'Age': data['age'], 'Gender': data['gender'],
        'SBP_raw': sbp, 'SBP_category': sbp_to_category(sbp),
        'DBP_raw': dbp, 'DBP_category': dbp_to_category(dbp),
        'BMI': bmi, 'HR': data['hr'], 'SpO2': spo2,
        'Glucose': glucose, 'HGB': hgb, 'WBC': wbc,
        'CVD_Probability': round(prob, 4), 'CVD_Prediction': label,
        'Model_Used': available_models.get(model_choice, model_choice),
        'Threshold': threshold, 'Scaler_Applied': 'scaler' in models,
    }
    if shap_vals is not None:
        for feat, val in shap_vals.items():
            result_dict[f'SHAP_{feat}'] = round(val, 5)

    csv = pd.DataFrame([result_dict]).to_csv(index=False).encode('utf-8-sig')
    col_dl, col_back, _ = st.columns([2, 2, 1])
    with col_dl:
        st.download_button("📥  Нәтижені CSV Жүктеу", data=csv,
                           file_name="cvd_result.csv", mime='text/csv')
    with col_back:
        st.markdown('<div class="btn-grey">', unsafe_allow_html=True)
        if st.button("🔄  Жаңа пациент (қайта өлшеу)", key="retry_btn", use_container_width=True):
            st.session_state.page = 'form'
            st.session_state.result_data = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <hr style='border-color:#d0d4e8; margin-top:40px;'>
    <div style='text-align:center; padding:20px; font-size:12px;'>
        <span style='color:#5a5a7a !important;'>
            🫀 CVD Тәуекел Болжаушы · Дипломдық жұмыс · ҚазҰУ · 2026<br>
            Әл-Фараби атындағы Қазақ Ұлттық Университеті ·
            Жасанды интеллект және Big Data кафедрасы
        </span><br>
        <span style='color:#a0a0c0 !important;'>
            LightGBM · CatBoost · XGBoost · Random Forest · Decision Tree · Autogluon
        </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# МАРШРУТИЗАТОР
# ══════════════════════════════════════════════════════
current_page = st.session_state.page

if current_page == 'warning':
    page_warning()
elif current_page == 'form':
    page_form(models, available_models, model_choice, threshold)
elif current_page == 'result':
    page_result(models, available_models)
