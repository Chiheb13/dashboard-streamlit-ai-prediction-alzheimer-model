import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "active_page" not in st.session_state:
    st.session_state.active_page = "Accueil"

dark = st.session_state.dark_mode

# ─────────────────────────────────────────────────────────────────────────────
# THEME TOKENS
# ─────────────────────────────────────────────────────────────────────────────
if dark:
    BG         = "#0d0f18"
    CARD       = "#161927"
    CARD2      = "#1e2235"
    BORDER     = "#2a2d45"
    TEXT       = "#e8ecf4"
    SUBTEXT    = "#8892a4"
    ACCENT     = "#6c63ff"
    ACCENT2    = "#a78bfa"
    SIDEBAR_BG = "#111320"
    FONT_COLOR = "#e8ecf4"
    GRID_COLOR = "#1e2235"
    CODE_BG    = "#1e2235"
else:
    BG         = "#f0f2f8"
    CARD       = "#ffffff"
    CARD2      = "#f7f8fc"
    BORDER     = "#dde1ee"
    TEXT       = "#1a1d2e"
    SUBTEXT    = "#5c6478"
    ACCENT     = "#5b52f5"
    ACCENT2    = "#7c6fea"
    SIDEBAR_BG = "#ffffff"
    FONT_COLOR = "#1a1d2e"
    GRID_COLOR = "#e8ecf4"
    CODE_BG    = "#eef0fc"

PLOT_BG  = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"

DIAG_COLORS = {"CN": "#22c55e", "LMCI": "#f97316", "AD": "#ef4444"}
MODEL_COLORS = {
    "Régression Logistique": "#6c63ff",
    "Arbre de Décision":     "#f97316",
    "Random Forest":         "#22c55e",
    "XGBoost":               "#a78bfa",
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, .stApp {{
    font-family: 'Inter', sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}
.stApp {{ background-color: {BG} !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.4rem !important; padding-bottom: 2rem !important; max-width: 1200px; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] > div:first-child {{ padding: 0 !important; }}

/* Metrics */
[data-testid="metric-container"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    transition: box-shadow 0.2s;
}}
[data-testid="metric-container"]:hover {{ box-shadow: 0 4px 16px rgba(108,99,255,0.12); }}
[data-testid="metric-container"] label {{
    color: {SUBTEXT} !important;
    font-size: 0.75rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
}}
[data-testid="stMetricDelta"] {{ font-size: 0.78rem !important; }}

/* Text */
h1, h2, h3, h4 {{ color: {TEXT} !important; font-family: 'Inter', sans-serif !important; font-weight: 700 !important; }}
p {{ color: {SUBTEXT}; }}
label {{ color: {TEXT} !important; font-weight: 500; }}

/* Dataframe */
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; border: 1px solid {BORDER}; }}

/* Expander */
[data-testid="stExpander"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
}}
[data-testid="stExpander"] summary {{ color: {TEXT} !important; }}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}

/* Custom classes */
.dash-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 16px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}}
.info-box {{
    background: {CODE_BG};
    border-left: 3px solid {ACCENT};
    border-radius: 0 10px 10px 0;
    padding: 11px 15px;
    margin: 10px 0;
    color: {SUBTEXT};
    font-size: 0.87rem;
    line-height: 1.65;
}}
.section-label {{
    padding: 14px 20px 4px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SUBTEXT};
}}
.sidebar-stat {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 20px;
    font-size: 0.82rem;
    color: {SUBTEXT};
}}
.sidebar-stat b {{ color: {TEXT}; font-weight: 600; }}

/* Nav buttons — override streamlit defaults in sidebar */
[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: {SUBTEXT} !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 9px 16px !important;
    margin: 1px 8px !important;
    width: calc(100% - 16px) !important;
    box-shadow: none !important;
    transition: background 0.15s, color 0.15s !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: {'#1e2235' if dark else '#eef0fc'} !important;
    color: {TEXT} !important;
    transform: none !important;
    box-shadow: none !important;
}}

/* Predict button */
.main-predict-btn .stButton > button {{
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2}) !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 12px !important;
    box-shadow: 0 4px 14px {ACCENT}44 !important;
}}
.main-predict-btn .stButton > button:hover {{
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}}

/* Prediction box */
.pred-box {{
    border-radius: 18px;
    padding: 32px 24px;
    text-align: center;
    margin: 8px 0;
    transition: transform 0.2s;
}}
.pred-CN   {{ background: {'#0f2d1a' if dark else '#dcfce7'}; border: 1.5px solid #22c55e; }}
.pred-LMCI {{ background: {'#2d1a06' if dark else '#ffedd5'}; border: 1.5px solid #f97316; }}
.pred-AD   {{ background: {'#2d0a0a' if dark else '#fee2e2'}; border: 1.5px solid #ef4444; }}

/* Theme toggle button */
.theme-toggle .stButton > button {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 20px !important;
    font-size: 0.8rem !important;
    padding: 5px 14px !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    width: auto !important;
    margin: 0 !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("alzheimer_clean.csv")

@st.cache_resource
def train_models(df):
    X_raw = df.drop(columns=['DX.bl', 'APOE Genotype'], errors='ignore')
    y     = df['DX.bl']
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    X     = pd.get_dummies(X_raw, drop_first=False)
    bc    = X.select_dtypes(include='bool').columns
    X[bc] = X[bc].astype(int)
    X     = X.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    defs = {
        "Régression Logistique": LogisticRegression(max_iter=1000, random_state=42),
        "Arbre de Décision":     DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest":         RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost":               XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                               eval_metric='mlogloss', random_state=42, verbosity=0),
    }
    trained, preds, accs = {}, {}, {}
    for name, m in defs.items():
        m.fit(X_train, y_train)
        yp = m.predict(X_test)
        trained[name] = m
        preds[name]   = yp
        accs[name]    = accuracy_score(y_test, yp)
    return trained, preds, accs, X, X_train, X_test, y_train, y_test, le

df = load_data()
trained, preds, accs, X, X_train, X_test, y_train, y_test, le = train_models(df)

def plot_layout(fig, height=360, **kw):
    fig.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font_color=FONT_COLOR, font_family="Inter",
        height=height,
        margin=dict(t=40, b=30, l=10, r=10),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color=FONT_COLOR),
        **kw,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
PAGES = [
    ("🏠", "Accueil"),
    ("📊", "Exploration des Données"),
    ("🔗", "Analyse de Corrélation"),
    ("🤖", "Entraînement des Modèles"),
    ("📈", "Comparaison des Modèles"),
    ("🔮", "Prédiction Patient"),
]

with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style="padding:22px 20px 14px;">
      <div style="font-size:1.5rem;font-weight:800;color:{TEXT};">
        🧠 <span style="background:linear-gradient(90deg,{ACCENT},{ACCENT2});
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AlzPredict</span>
      </div>
      <div style="color:{SUBTEXT};font-size:0.75rem;margin-top:3px;letter-spacing:.03em;">
        Machine Learning · ADNI Dataset
      </div>
    </div>
    <div style="height:1px;background:{BORDER};margin:0 0 8px;"></div>
    """, unsafe_allow_html=True)

    # Theme toggle
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<div style='padding:6px 4px;font-size:0.8rem;color:{SUBTEXT};'>"
                    f"{'🌙 Mode Sombre' if dark else '☀️ Mode Clair'}</div>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="theme-toggle">', unsafe_allow_html=True)
        if st.button("⇄"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"<div class='section-label'>Pages</div>", unsafe_allow_html=True)

    for icon, label in PAGES:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.active_page = label
            st.rerun()

    # Dataset stats
    st.markdown(f"""
    <div style="height:1px;background:{BORDER};margin:14px 0 6px;"></div>
    <div class="section-label">Dataset ADNI</div>
    """, unsafe_allow_html=True)

    for icon, lbl, val in [
        ("👥","Patients", len(df)),
        ("📋","Features", len(X.columns)),
        ("🎯","Classes", "CN · LMCI · AD"),
        ("✂️","Split", "80% Train / 20% Test"),
    ]:
        st.markdown(f"""
        <div class="sidebar-stat">
          <span style="color:{SUBTEXT};">{icon} {lbl}</span>
          <b>{val}</b>
        </div>""", unsafe_allow_html=True)

    # Model perf
    st.markdown(f"""
    <div style="height:1px;background:{BORDER};margin:10px 0 6px;"></div>
    <div class="section-label">Performances</div>
    """, unsafe_allow_html=True)

    for name, acc in sorted(accs.items(), key=lambda x: -x[1]):
        short = {"Régression Logistique":"Log. Regression",
                 "Arbre de Décision":"Decision Tree",
                 "Random Forest":"Random Forest",
                 "XGBoost":"XGBoost"}[name]
        c = MODEL_COLORS[name]
        bar_w = int(acc * 80)
        st.markdown(f"""
        <div style="padding:4px 20px;">
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:2px;">
            <span style="color:{c};font-weight:600;">{short}</span>
            <b style="color:{TEXT};">{acc:.1%}</b>
          </div>
          <div style="height:4px;background:{BORDER};border-radius:4px;">
            <div style="height:4px;width:{bar_w}%;background:{c};border-radius:4px;"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
page = st.session_state.active_page

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ═════════════════════════════════════════════════════════════════════════════
if page == "Accueil":
    st.markdown(f"<h1 style='margin-bottom:2px;'>🧠 Prédiction de la Maladie d'Alzheimer</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SUBTEXT};margin-top:0;'>Projet Machine Learning — Dataset ADNI · 627 patients · 4 modèles comparés</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    best_name = max(accs, key=accs.get)
    c1.metric("👥 Patients", "627", "ADNI dataset")
    c2.metric("🧬 Features", len(X.columns), "après encoding")
    c3.metric("🏆 Meilleure Accuracy", f"{accs[best_name]:.1%}", best_name)
    c4.metric("⚕️ Classes cibles", "3", "CN · LMCI · AD")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown(f"""
        <div class="dash-card">
          <h3 style="margin-top:0;">🎯 Objectif du Projet</h3>
          <p>Développer et comparer des modèles ML pour prédire le <b style='color:{TEXT};'>diagnostic d'Alzheimer</b>
          parmi 3 classes :</p>
          <div class="info-box" style="border-left-color:#22c55e;">
            🟢 <b>CN</b> — Cognitively Normal : aucune maladie détectée
          </div>
          <div class="info-box" style="border-left-color:#f97316;">
            🟠 <b>LMCI</b> — Mild Cognitive Impairment : stade intermédiaire
          </div>
          <div class="info-box" style="border-left-color:#ef4444;">
            🔴 <b>AD</b> — Alzheimer's Disease : maladie diagnostiquée
          </div>
          <p style="margin-top:14px;font-size:0.85rem;">
            <b style="color:{TEXT};">Variables clés :</b>
            Score MMSE, génotype APOE4, âge, éducation, ethnie, race.
          </p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="dash-card">
          <h3 style="margin-top:0;">🧰 Résultats des Modèles</h3>
          <table style='width:100%;border-collapse:collapse;font-size:0.86rem;'>
            <tr style='border-bottom:1px solid {BORDER};'>
              <th style='text-align:left;padding:8px 4px;color:{SUBTEXT};font-weight:600;'>Modèle</th>
              <th style='text-align:left;padding:8px 4px;color:{SUBTEXT};font-weight:600;'>Principe</th>
              <th style='text-align:center;padding:8px 4px;color:{SUBTEXT};font-weight:600;'>Accuracy</th>
            </tr>""" + "".join(f"""
            <tr style='border-bottom:1px solid {BORDER};'>
              <td style='padding:9px 4px;font-weight:700;color:{MODEL_COLORS[n]};'>{n}</td>
              <td style='padding:9px 4px;color:{SUBTEXT};'>{"Frontières linéaires" if "Logistique" in n else "Questions successives" if "Décision" in n else "Vote de 100 arbres" if "Forest" in n else "Correction séquentielle"}</td>
              <td style='padding:9px 4px;text-align:center;font-weight:800;color:{TEXT};'>{accs[n]:.1%}</td>
            </tr>""" for n in accs) + f"""
          </table>
        </div>""", unsafe_allow_html=True)

    with col_right:
        counts = df['DX.bl'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=counts.index, values=counts.values,
            hole=0.52,
            marker_colors=[DIAG_COLORS[c] for c in counts.index],
            textinfo='label+percent',
            textfont_size=12,
        ))
        fig_pie.update_layout(
            title="Répartition des patients",
            paper_bgcolor=PAPER_BG, font_color=FONT_COLOR,
            font_family="Inter", height=290,
            margin=dict(t=40, b=10, l=10, r=10), showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown(f"""
        <div class="dash-card" style="text-align:center;margin-top:0;">
          <div style='color:{SUBTEXT};font-size:0.72rem;text-transform:uppercase;letter-spacing:.1em;font-weight:600;'>
            🏆 Meilleur Modèle
          </div>
          <div style='font-size:1.2rem;font-weight:800;
            background:linear-gradient(90deg,{ACCENT},{ACCENT2});
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            margin:8px 0 4px;'>{best_name}</div>
          <div style='font-size:2.4rem;font-weight:900;color:{TEXT};line-height:1;'>
            {accs[best_name]:.1%}
          </div>
          <div style='color:{SUBTEXT};font-size:0.78rem;margin-top:6px;'>accuracy sur données de test</div>
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Exploration des Données":
    st.markdown("<h1>📊 Exploration des Données</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SUBTEXT};'>Analyse visuelle du dataset avant modélisation.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("👀 Aperçu du dataset", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", len(df))
    c2.metric("Variables", len(df.columns))
    c3.metric("Valeurs manquantes", int(df.isnull().sum().sum()))
    st.markdown("<br>", unsafe_allow_html=True)

    # 1 — Target distribution (two separate charts)
    st.markdown("<h3>1️⃣ Distribution de la Target</h3>", unsafe_allow_html=True)
    counts = df['DX.bl'].value_counts()
    col1, col2 = st.columns(2)

    with col1:
        fig_bar = go.Figure(go.Bar(
            x=list(counts.index), y=list(counts.values),
            marker_color=[DIAG_COLORS[c] for c in counts.index],
            text=list(counts.values), textposition='outside',
        ))
        plot_layout(fig_bar, height=310, title="Nombre de patients", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_pie2 = go.Figure(go.Pie(
            labels=list(counts.index), values=list(counts.values),
            hole=0.45,
            marker_colors=[DIAG_COLORS[c] for c in counts.index],
            textinfo='label+percent', textfont_size=12,
        ))
        fig_pie2.update_layout(
            title="Proportion", paper_bgcolor=PAPER_BG,
            font_color=FONT_COLOR, font_family="Inter",
            height=310, margin=dict(t=40, b=10), showlegend=False,
        )
        st.plotly_chart(fig_pie2, use_container_width=True)

    st.markdown("""<div class="info-box">
    📌 Le dataset est <b>déséquilibré</b> : LMCI est majoritaire (~48%), puis CN (~30%) et AD (~21%).
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # 2 — MMSE
    st.markdown("<h3>2️⃣ Score MMSE par Diagnostic</h3>", unsafe_allow_html=True)
    fig_box = px.box(df, x='DX.bl', y='MMSE', color='DX.bl',
                     color_discrete_map=DIAG_COLORS,
                     category_orders={"DX.bl": ["CN","LMCI","AD"]})
    plot_layout(fig_box, height=340, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown("""<div class="info-box">
    📌 Le MMSE <b>diminue avec la sévérité</b> : CN ≈ 28, LMCI ≈ 27, AD ≈ 23. C'est le meilleur prédicteur.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # 3 & 4 — Age + APOE4
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3>3️⃣ Âge par Diagnostic</h3>", unsafe_allow_html=True)
        fig_vio = px.violin(df, x='DX.bl', y='AGE', color='DX.bl',
                            color_discrete_map=DIAG_COLORS,
                            category_orders={"DX.bl":["CN","LMCI","AD"]}, box=True)
        plot_layout(fig_vio, height=320, showlegend=False)
        st.plotly_chart(fig_vio, use_container_width=True)

    with col2:
        st.markdown("<h3>4️⃣ APOE4 par Diagnostic</h3>", unsafe_allow_html=True)
        apoe = pd.crosstab(df['DX.bl'], df['APOE4']).reset_index()
        apoe_m = apoe.melt(id_vars='DX.bl', var_name='APOE4', value_name='count')
        fig_apoe = px.bar(apoe_m, x='DX.bl', y='count',
                          color='APOE4', barmode='group',
                          color_continuous_scale='Blues')
        plot_layout(fig_apoe, height=320)
        st.plotly_chart(fig_apoe, use_container_width=True)

    st.markdown("---")

    # 5 — Categorical
    st.markdown("<h3>5️⃣ Variables Catégorielles</h3>", unsafe_allow_html=True)
    cat_cols = [c for c in ['PTETHCAT','PTRACCAT','imputed_genotype'] if c in df.columns]
    chosen = st.selectbox("Variable :", cat_cols)
    fig_cat = px.histogram(df, x=chosen, color='DX.bl',
                           color_discrete_map=DIAG_COLORS, barmode='group')
    plot_layout(fig_cat, height=340)
    st.plotly_chart(fig_cat, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CORRÉLATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Analyse de Corrélation":
    st.markdown("<h1>🔗 Analyse de Corrélation</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SUBTEXT};'>Relations entre variables et influence sur le diagnostic.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    df_c = df.copy()
    df_c['DX_num'] = df_c['DX.bl'].map({'CN':0,'LMCI':1,'AD':2})
    df_c = df_c.drop(columns=['DX.bl'])
    df_enc = pd.get_dummies(df_c, drop_first=False)
    bc = df_enc.select_dtypes(include='bool').columns
    df_enc[bc] = df_enc[bc].astype(int)
    df_enc = df_enc.astype(float)

    st.markdown("<h3>1️⃣ Matrice de Corrélation</h3>", unsafe_allow_html=True)
    corr = df_enc.corr()
    fig_h = px.imshow(corr, text_auto=".2f",
                      color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig_h.update_layout(paper_bgcolor=PAPER_BG, font_color=FONT_COLOR,
                        font_family="Inter", height=480,
                        margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("""<div class="info-box">
    📌 <b>APOE Genotype</b> et <b>APOE4</b> sont très corrélées → on retire APOE Genotype
    pour éviter la <b>multicolinéarité</b>.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<h3>2️⃣ Corrélation avec la Target (DX)</h3>", unsafe_allow_html=True)
    ct = df_enc.corr()[['DX_num']].drop('DX_num').sort_values('DX_num', ascending=True)
    fig_ct = px.bar(ct.reset_index(), x='DX_num', y='index',
                    orientation='h',
                    color='DX_num', color_continuous_scale='RdBu_r',
                    range_color=[-1,1],
                    labels={"DX_num":"Corrélation","index":"Variable"})
    plot_layout(fig_ct, height=400)
    st.plotly_chart(fig_ct, use_container_width=True)

    st.markdown("""<div class="info-box">
    📌 <b>MMSE</b> a la corrélation négative la plus forte (MMSE bas = sévérité élevée).
    <b>APOE4</b> est positivement corrélée (plus d'allèles = plus de risque).
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ENTRAÎNEMENT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Entraînement des Modèles":
    st.markdown("<h1>🤖 Entraînement des Modèles</h1>", unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("Train (80%)", f"{len(X_train)} patients", "Pour apprendre")
    c2.metric("Test (20%)",  f"{len(X_test)} patients",  "Pour évaluer")
    c3.metric("Features",    len(X.columns))

    st.markdown("""<div class="info-box">
    📌 On entraîne sur <b>80%</b> des données puis on évalue sur les <b>20% restants</b>
    — des données jamais vues pendant l'entraînement.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    selected = st.selectbox("🔍 Choisir un modèle :", list(trained.keys()))
    model  = trained[selected]
    y_pred = preds[selected]
    acc    = accs[selected]
    color  = MODEL_COLORS[selected]

    st.markdown(f"<h3>📋 Résultats — {selected}</h3>", unsafe_allow_html=True)
    st.metric("Accuracy", f"{acc:.1%}")

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    rep_df = pd.DataFrame(report).T
    rep_df = rep_df.loc[list(le.classes_), ['precision','recall','f1-score','support']].round(3)
    st.dataframe(rep_df, use_container_width=True)

    st.markdown(f"""<div class="info-box">
    📌 <b>Precision</b> = parmi les prédictions positives, combien sont vraiment positives<br>
    📌 <b>Recall</b> = parmi les vrais positifs, combien a-t-on bien détectés<br>
    📌 <b>F1-Score</b> = équilibre entre precision et recall
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3>🔲 Matrice de Confusion</h3>", unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm,
                           x=list(le.classes_), y=list(le.classes_),
                           text_auto=True,
                           color_continuous_scale=[[0, CARD2],[1, color]],
                           labels=dict(x="Prédit", y="Réel"))
        fig_cm.update_layout(paper_bgcolor=PAPER_BG, font_color=FONT_COLOR,
                             font_family="Inter", height=320,
                             margin=dict(t=30, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("""<div class="info-box">📌 La diagonale = bonnes prédictions.</div>""",
                    unsafe_allow_html=True)

    with col2:
        st.markdown("<h3>📊 Top 10 Features</h3>", unsafe_allow_html=True)
        if hasattr(model, 'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=X.columns)
        else:
            imp = pd.Series(np.abs(model.coef_).mean(axis=0), index=X.columns)
        top10 = imp.sort_values(ascending=True).tail(10)
        fig_fi = px.bar(top10.reset_index(), x=top10.values, y='index',
                        orientation='h', color_discrete_sequence=[color],
                        labels={"x":"Importance","index":""})
        plot_layout(fig_fi, height=320)
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    infos = {
        "Régression Logistique": ("🔵","Coefficients linéaires par feature","max_iter=1000","Rapide, interprétable","Relations non-linéaires mal gérées"),
        "Arbre de Décision":     ("🟠","Questions successives (seuils)","max_depth=4","Très visuel, intuitif","Overfitting possible"),
        "Random Forest":         ("🟢","Vote de 100 arbres aléatoires","n_estimators=100, max_depth=5","Robuste, performant","Moins interprétable"),
        "XGBoost":               ("🟣","Arbres séquentiels correcteurs","lr=0.1, n_estimators=100","Très haute performance","Complexe à paramétrer"),
    }
    icon, desc, params, pros, cons = infos[selected]
    st.markdown(f"""
    <div class="dash-card">
      <h3 style="margin-top:0;">{icon} Comment fonctionne {selected} ?</h3>
      <p style="color:{SUBTEXT};">{desc}</p>
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;">
        <div class="info-box" style="flex:1;min-width:140px;">⚙️ <b>Params</b><br><code>{params}</code></div>
        <div class="info-box" style="flex:1;min-width:140px;border-left-color:#22c55e;">✅ <b>Avantages</b><br>{pros}</div>
        <div class="info-box" style="flex:1;min-width:140px;border-left-color:#ef4444;">❌ <b>Limites</b><br>{cons}</div>
      </div>
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — COMPARAISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Comparaison des Modèles":
    st.markdown("<h1>📈 Comparaison des 4 Modèles</h1>", unsafe_allow_html=True)
    st.markdown("---")

    acc_df = (pd.DataFrame({"Modèle": list(accs.keys()), "Accuracy": list(accs.values())})
              .sort_values("Accuracy", ascending=False)
              .reset_index(drop=True))
    acc_df.index += 1
    best_acc = acc_df['Accuracy'].max()

    fig_acc = px.bar(acc_df, x='Modèle', y='Accuracy', color='Modèle',
                     color_discrete_map=MODEL_COLORS,
                     text=acc_df['Accuracy'].map(lambda x: f"{x:.1%}"),
                     range_y=[0, 1])
    fig_acc.add_hline(y=best_acc, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Meilleur : {best_acc:.1%}",
                      annotation_position="top right",
                      annotation_font_color=FONT_COLOR)
    fig_acc.update_traces(textposition='outside')
    plot_layout(fig_acc, height=370, showlegend=False)
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("<h3>📋 Tableau Récapitulatif</h3>", unsafe_allow_html=True)
    disp = acc_df.copy()
    disp['Accuracy (%)'] = disp['Accuracy'].map(lambda x: f"{x:.1%}")
    disp['Rang'] = range(1, len(disp)+1)
    st.dataframe(disp[['Rang','Modèle','Accuracy (%)']], use_container_width=True, hide_index=True)
    st.markdown("---")

    # Feature importance — 4 separate charts (fixed bug: no make_subplots)
    st.markdown("<h3>🏆 Top 5 Features par Modèle</h3>", unsafe_allow_html=True)
    col_pairs = st.columns(2)
    for i, (model_name, model) in enumerate(trained.items()):
        if hasattr(model, 'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=X.columns)
        else:
            imp = pd.Series(np.abs(model.coef_).mean(axis=0), index=X.columns)
        top5 = imp.sort_values(ascending=True).tail(5)
        color = MODEL_COLORS[model_name]
        with col_pairs[i % 2]:
            fig_fi = px.bar(top5.reset_index(), x=top5.values, y='index',
                            orientation='h', color_discrete_sequence=[color],
                            title=model_name, labels={"x":"Importance","index":""})
            plot_layout(fig_fi, height=240)
            fig_fi.update_layout(margin=dict(t=42, b=10, l=10, r=10))
            st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3>📚 Synthèse Comparative</h3>", unsafe_allow_html=True)
    synth = pd.DataFrame({
        'Modèle':          list(accs.keys()),
        'Accuracy':        [f"{accs[n]:.1%}" for n in accs],
        'Avantages':       ['Rapide, interprétable','Très visuel','Robuste, précis','Très performant'],
        'Inconvénients':   ['Moins performant','Overfitting possible','Boîte noire','Complexe'],
        'Usage':           ['Baseline','Explication','Production','Compétitions ML'],
    })
    st.dataframe(synth, use_container_width=True, hide_index=True)

    best_name = acc_df.iloc[0]['Modèle']
    st.markdown(f"""<div class="info-box">
    🏆 <b>Meilleur modèle : {best_name}</b> avec <b>{best_acc:.1%}</b> d'accuracy.<br>
    📌 <b>MMSE</b> et <b>APOE4</b> sont les features les plus importantes dans tous les modèles.<br>
    📌 La classe <b>LMCI</b> est la plus difficile (stade intermédiaire).
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PRÉDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Prédiction Patient":
    st.markdown("<h1>🔮 Prédiction pour un Nouveau Patient</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{SUBTEXT};'>Saisissez les caractéristiques du patient pour obtenir un diagnostic prédit.</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown(f"<div class='dash-card'><h3 style='margin-top:0;'>📝 Informations Patient</h3>",
                    unsafe_allow_html=True)
        age       = st.slider("🎂 Âge", 55, 90, 75)
        mmse      = st.slider("🧪 Score MMSE  (0 = très sévère  ·  30 = normal)", 0, 30, 27)
        education = st.slider("🎓 Années d'éducation", 4, 20, 14)
        apoe4     = st.selectbox("🧬 APOE4 (allèles de risque)", [0,1,2],
                                 format_func=lambda x: {0:"0 — Aucun risque",1:"1 — Risque modéré",2:"2 — Risque élevé"}[x])
        gender    = st.selectbox("⚥ Genre", [("Femme",0),("Homme",1)],
                                 format_func=lambda x: x[0])
        ethnicity = st.selectbox("🌍 Ethnicité", df['PTETHCAT'].unique().tolist())
        race      = st.selectbox("🌐 Race", df['PTRACCAT'].unique().tolist())
        imputed   = st.checkbox("🔬 Génotype imputé indirectement", False)
        chosen_m  = st.selectbox("🤖 Modèle", list(trained.keys()))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="main-predict-btn">', unsafe_allow_html=True)
        predict = st.button("🔮 Prédire le Diagnostic", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown("<h3>📊 Résultat</h3>", unsafe_allow_html=True)

        if predict:
            inp = pd.DataFrame([{
                'AGE': age, 'PTGENDER': gender[1], 'PTEDUCAT': education,
                'PTETHCAT': ethnicity, 'PTRACCAT': race, 'APOE4': apoe4,
                'MMSE': mmse, 'imputed_genotype': imputed,
            }])
            inp_enc = pd.get_dummies(inp)
            bc2 = inp_enc.select_dtypes(include='bool').columns
            inp_enc[bc2] = inp_enc[bc2].astype(int)
            for col in X.columns:
                if col not in inp_enc.columns:
                    inp_enc[col] = 0
            inp_enc = inp_enc[X.columns].astype(float)

            m        = trained[chosen_m]
            pred_i   = m.predict(inp_enc)[0]
            pred_lbl = le.inverse_transform([pred_i])[0]

            icons     = {"CN":"🟢","LMCI":"🟠","AD":"🔴"}
            label_fr  = {"CN":"Cognitively Normal","LMCI":"Mild Cognitive Impairment","AD":"Alzheimer's Disease"}
            interps   = {
                "CN":   f"✅ Profil cognitif <b>normal</b>. Aucun signe d'Alzheimer détecté.",
                "LMCI": f"⚠️ Signes de <b>déficience cognitive légère</b>. Suivi médical recommandé.",
                "AD":   f"🚨 Profil compatible avec la <b>maladie d'Alzheimer</b>. Consultation spécialisée nécessaire.",
            }
            diag_col = DIAG_COLORS[pred_lbl]

            st.markdown(f"""
            <div class="pred-box pred-{pred_lbl}">
              <div style='font-size:3.5rem;line-height:1;'>{icons[pred_lbl]}</div>
              <div style='font-size:2.2rem;font-weight:900;color:{TEXT};margin:8px 0 4px;'>{pred_lbl}</div>
              <div style='color:{SUBTEXT};font-size:0.9rem;'>{label_fr[pred_lbl]}</div>
              <div style='margin-top:10px;'>
                <span style='background:{diag_col}22;color:{diag_col};padding:3px 12px;border-radius:20px;font-size:0.78rem;font-weight:600;'>
                  via {chosen_m}
                </span>
              </div>
            </div>""", unsafe_allow_html=True)

            if hasattr(m, 'predict_proba'):
                proba = m.predict_proba(inp_enc)[0]
                fig_p = px.bar(
                    x=list(le.classes_), y=proba,
                    color=list(le.classes_),
                    color_discrete_map=DIAG_COLORS,
                    text=[f"{v:.1%}" for v in proba],
                    range_y=[0,1],
                    labels={"x":"Classe","y":"Probabilité"},
                )
                fig_p.update_traces(textposition='outside')
                plot_layout(fig_p, height=270, showlegend=False,
                            title="Probabilité par classe")
                st.plotly_chart(fig_p, use_container_width=True)

            st.markdown(f"""<div class="info-box" style="border-left-color:{diag_col};">
            {interps[pred_lbl]}
            </div>""", unsafe_allow_html=True)

            st.markdown("""<div class="info-box" style="border-left-color:#f59e0b;">
            ⚠️ <b>Avertissement</b> : Ce résultat est statistique et
            <b>ne remplace pas un diagnostic médical professionnel</b>.
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="background:{CARD};border:2px dashed {BORDER};border-radius:16px;
                        padding:60px 20px;text-align:center;">
              <div style='font-size:4rem;'>🔮</div>
              <div style='color:{SUBTEXT};margin-top:12px;'>
                Remplissez le formulaire et cliquez sur<br>
                <b style='color:{ACCENT};'>Prédire le Diagnostic</b>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4>📏 Référence MMSE</h4>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Score':       ['27–30','21–26','10–20','< 10'],
            'Niveau':      ['Normal','Léger','Modéré','Sévère'],
            'Risque AD':   ['Faible','Possible','Probable','Très élevé'],
        }), hide_index=True, use_container_width=True)
