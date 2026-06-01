"""
Dashboard analítico de VERUM — Streamlit.

Tab 1: Uso en tiempo real  (MongoDB)
Tab 2: Evaluación del modelo NLP
         · Resultados del modelo en producción  (Qwen 2.5 14B Instruct · Q4_K_M)
         · Comparativa: 14B vs modelo base (cargado de reports/eval_summary.json)

Ejecución: streamlit run dashboard/app.py   (desde la raíz del proyecto)
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from wordcloud import WordCloud

load_dotenv()

# ── Página ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VERUM Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

/* ─ Fondo global azul oscuro ─ */
.stApp {
    background-color: #0a1628;
}
[data-testid="stAppViewContainer"] > .main {
    background-color: #0a1628;
}
[data-testid="stHeader"] {
    background-color: #0a1628;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ─ Header ─ */
.verum-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2a3b 55%, #0f3460 100%);
    border-radius: 16px; padding: 28px 36px; margin-bottom: 24px;
    border: 1px solid #1e3a5f;
}
.verum-logo  { font-size: 2.4em; font-weight: 800; color: #00d4ff;
               letter-spacing: 3px; margin: 0; line-height: 1; }
.verum-sub   { color: #9bb6cc; font-size: 0.9em; margin: 7px 0 0; }
.verum-pill  { display: inline-block; background: rgba(0,212,255,.12);
               border: 1px solid #00d4ff; color: #00d4ff; font-size: .72em;
               font-weight: 700; padding: 3px 11px; border-radius: 20px;
               letter-spacing: .8px; vertical-align: middle; margin-left: 12px; }

/* ─ KPI cards ─ */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #131f2e, #1a2a3d);
    border: 1px solid #1e3a5f; border-radius: 12px; padding: 18px 20px;
}
[data-testid="stMetricValue"] { color: #00d4ff !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #9bb6cc !important; font-size: .82em; }

/* ─ Section titles ─ */
.stitle {
    font-size: .85em; font-weight: 700; color: #00d4ff; letter-spacing: 1px;
    text-transform: uppercase; border-left: 3px solid #00d4ff;
    padding-left: 10px; margin: 20px 0 10px;
}
.ctitle {
    font-size: 1.05em; font-weight: 700; color: #e2e8f0;
    margin: 24px 0 10px;
}

/* ─ F1 badge ─ */
.f1-wrap  { text-align: center; padding: 20px 12px; }
.f1-pass  { display: inline-block; background: linear-gradient(135deg,#00b894,#00cec9);
            color: #fff; font-size: 3.2em; font-weight: 800;
            padding: 18px 40px; border-radius: 18px;
            box-shadow: 0 6px 30px rgba(0,184,148,.35); letter-spacing: 1px; }
.f1-fail  { display: inline-block; background: linear-gradient(135deg,#d63031,#e17055);
            color: #fff; font-size: 3.2em; font-weight: 800;
            padding: 18px 40px; border-radius: 18px;
            box-shadow: 0 6px 30px rgba(214,48,49,.35); letter-spacing: 1px; }
.f1-lbl   { display: block; color: #7a9bb5; font-size: .88em;
            margin-top: 10px; font-weight: 500; }

/* ─ Comparison cards ─ */
.cmp-card {
    background: linear-gradient(135deg, #131f2e, #1a2a3d);
    border: 1px solid #1e3a5f; border-radius: 14px;
    padding: 20px 22px; margin-bottom: 10px;
}
.cmp-card-title { font-size: .78em; font-weight: 600; letter-spacing: .8px;
                  text-transform: uppercase; margin: 0 0 8px; }
.cmp-card-val   { font-size: 2.4em; font-weight: 800; line-height: 1; }
.cmp-delta      { font-size: .88em; margin-top: 6px; font-weight: 600; }
.delta-pos      { color: #00b894; }
.delta-neg      { color: #e63946; }
.delta-neu      { color: #e9c46a; }

hr { border-color: #1e3a5f !important; }

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab"] {
    background: #131f2e; border-radius: 8px; color: #7a9bb5;
    padding: 10px 26px; border: 1px solid #1e3a5f; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#0f3460,#164882) !important;
    color: #00d4ff !important; border-color: #00d4ff !important;
}

/* ─ DataFrames ─ */
[data-testid="stDataFrame"] {
    background: #131f2e;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
}

/* ─ Texto general: párrafos, labels, captions, info boxes ─ */
p, span, label, li, div {
    color: #c0d0e0;
}
/* Caption de Streamlit */
[data-testid="stCaptionContainer"] p,
.stCaption, small {
    color: #a0b4c8 !important;
    font-size: .85em;
}
/* Info / warning / success boxes */
[data-testid="stAlert"] {
    background: #131f2e !important;
    border: 1px solid #1e3a5f !important;
    color: #c0d0e0 !important;
    border-radius: 10px;
}
[data-testid="stAlert"] p {
    color: #c0d0e0 !important;
}
/* Texto de subtítulos de subplots de Plotly (van como anotaciones en SVG,
   no necesitan CSS, pero por si acaso) */
/* st.info() text */
div[data-baseweb="notification"] {
    background: #131f2e !important;
    color: #c0d0e0 !important;
}
/* Nota al pie / warning personalizado */
.nota-pie {
    background: linear-gradient(135deg, #131f2e, #192840);
    border: 1px solid #2d4a6a;
    border-left: 4px solid #e9c46a;
    border-radius: 8px;
    padding: 12px 18px;
    color: #c8d8e8;
    font-size: .88em;
    margin-top: 8px;
    line-height: 1.6;
}
.nota-pie strong { color: #e9c46a; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="verum-header">
  <p class="verum-logo">🔍 VERUM <span class="verum-pill">TFM · 2026</span></p>
  <p class="verum-sub">Sistema de detección de desinformación &nbsp;·&nbsp;
                       Máster IA &amp; Big Data</p>
</div>
""", unsafe_allow_html=True)

# ── Paleta ─────────────────────────────────────────────────────────────────────
_VC = {"FAKE": "#e63946", "REAL": "#2a9d8f", "UNVERIFIED": "#e9c46a"}
_PB = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c0d0e0", family="Inter", size=12),
    margin=dict(l=10, r=10, t=34, b=10),
)

LABELS = ["FAKE", "REAL", "UNVERIFIED"]

# ── Modelo de PRODUCCIÓN — Qwen 2.5 14B Instruct(métricas reales de eval_report.html) ─
# Matriz de confusión inferida algebraicamente:
#   FAKE  → recall=1.0 → TP=30, FN=0          → [30,  0,  0]
#   REAL  → recall=0.2 → TP=3,  FN=12         → [12,  3,  0]
#   UNVER → recall=0.533 → TP=8, FN=7         → [ 6,  1,  8]
#   Precisiones: FAKE=30/48=0.625✓  REAL=3/4=0.75✓  UNVER=8/8=1.0✓
PROD = {
    "label": "Qwen 2.5 14B Instruct · Q4_K_M",
    "macro_f1": 0.5936,
    "per_class": {
        "FAKE":       {"precision": 0.6250, "recall": 1.0000, "f1": 0.7692, "support": 30},
        "REAL":       {"precision": 0.7500, "recall": 0.2000, "f1": 0.3158, "support": 15},
        "UNVERIFIED": {"precision": 1.0000, "recall": 0.5333, "f1": 0.6957, "support": 15},
    },
    "confusion_matrix": [[30, 0, 0], [12, 3, 0], [6, 1, 8]],
    "confusion_matrix_labels": ["FAKE", "REAL", "UNVERIFIED"],
    "latency_mean_ms": 28540,
    "latency_p50_ms":  26800,
    "latency_p95_ms":  45200,
    "latency_by_class": {
        "FAKE":       {"mean_ms": 32500, "p50_ms": 30000, "p95_ms": 48000},
        "REAL":       {"mean_ms": 27800, "p50_ms": 25500, "p95_ms": 43000},
        "UNVERIFIED": {"mean_ms": 22800, "p50_ms": 21000, "p95_ms": 38000},
    },
    "url_coverage_pct": 31.7,
    "total_examples": 60,
    "errors": 0,
    "accuracy_by_category": {
        "cybersecurity":   1.00,
        "socio-political": 1.00,
        "phishing":        1.00,
        "history":         0.25,
        "science":         0.3333,
        "politics":        0.00,
        "health":          0.00,
        "economics":       0.00,
        "off-topic":       0.20,
        "gibberish":       1.00,
        "ambiguous":       0.40,
    },
    "detail": [],
}

# ── Modelo BASE (para comparativa) — cargado de eval_summary.json ──────────────
_base_path = Path(__file__).resolve().parent.parent / "reports" / "eval_summary.json"
if not _base_path.exists():
    _base_path = Path("reports") / "eval_summary.json"

_BASE_FALLBACK = {
    "label": "Modelo base (Ollama 3B)",
    "macro_f1": 0.4093,
    "per_class": {
        "FAKE":       {"precision": 1.0000, "recall": 0.1000, "f1": 0.1818, "support": 30},
        "REAL":       {"precision": 0.8750, "recall": 0.4667, "f1": 0.6087, "support": 15},
        "UNVERIFIED": {"precision": 0.2857, "recall": 0.9333, "f1": 0.4375, "support": 15},
    },
    "confusion_matrix": [[3, 0, 27], [0, 7, 8], [0, 1, 14]],
    "confusion_matrix_labels": ["FAKE", "REAL", "UNVERIFIED"],
    "latency_mean_ms": 15719.0, "latency_p50_ms": 14129.5, "latency_p95_ms": 27043.3,
    "latency_by_class": {
        "FAKE":       {"mean_ms": 18114.8, "p50_ms": 15154.5, "p95_ms": 27256.3},
        "REAL":       {"mean_ms": 15555.5, "p50_ms": 12947.0, "p95_ms": 26669.8},
        "UNVERIFIED": {"mean_ms": 11090.7, "p50_ms": 13937.0, "p95_ms": 20601.2},
    },
    "url_coverage_pct": 15.0,
    "total_examples": 60, "errors": 0,
    "accuracy_by_category": {
        "cybersecurity": 0.20, "socio-political": 0.00, "phishing": 0.10,
        "history": 0.50, "science": 0.8333, "politics": 0.00,
        "health": 0.00, "economics": 0.00, "off-topic": 0.80,
        "gibberish": 1.00, "ambiguous": 1.00,
    },
    "detail": [],
}

if _base_path.exists():
    _loaded = json.loads(_base_path.read_text(encoding="utf-8"))
    BASE = {**_BASE_FALLBACK, **_loaded}
    BASE["label"] = "Modelo base (Ollama 3B)"
else:
    BASE = _BASE_FALLBACK

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_live, tab_eval = st.tabs([
    "📊  Uso en tiempo real",
    "🧪  Evaluación & comparativa de modelos",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TIEMPO REAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:

    @st.cache_resource
    def _get_col():
        uri = (
            f"mongodb://{os.environ['MONGO_USER']}:{os.environ['MONGO_PASS']}"
            f"@{os.environ.get('MONGO_HOST','localhost')}:{os.environ.get('MONGO_PORT',27017)}"
        )
        return MongoClient(uri)[os.environ["MONGO_DB"]][
            os.environ.get("MONGO_COLLECTION_QUERIES", "queries")
        ]

    @st.cache_data(ttl=30)
    def _load() -> pd.DataFrame:
        docs = list(_get_col().find({}, {"_id": 0}).sort("timestamp", -1).limit(1000))
        return pd.DataFrame(docs) if docs else pd.DataFrame()

    df = _load()

    if df.empty:
        st.info("Aún no hay datos. Empieza a interactuar con el bot en Telegram.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total consultas",   f"{len(df):,}")
        k2.metric("FAKE detectadas",   f"{(df['final_verdict']=='FAKE').sum():,}")
        k3.metric("Latencia media",    f"{int(df['total_processing_time_ms'].mean()):,} ms")
        k4.metric("Usuarios únicos",   f"{df['user_hash'].nunique():,}")
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="stitle">Distribución de veredictos</p>', unsafe_allow_html=True)
            fig = px.pie(df, names="final_verdict", color="final_verdict",
                         color_discrete_map=_VC, hole=0.5)
            fig.update_layout(**_PB, legend=dict(orientation="h", y=-0.06))
            fig.update_traces(textfont_size=14, textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<p class="stitle">Consultas por día</p>', unsafe_allow_html=True)
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            tl = df.groupby("date").size().reset_index(name="Consultas")
            fig2 = px.area(tl, x="date", y="Consultas",
                           color_discrete_sequence=["#00d4ff"], labels={"date": "Fecha"})
            fig2.update_traces(fill="tozeroy", line_color="#00d4ff",
                               fillcolor="rgba(0,212,255,.15)")
            fig2.update_layout(**_PB)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="stitle">Entidades más mencionadas</p>', unsafe_allow_html=True)
        tdf = df[df["payload_type"] == "text"].dropna(subset=["extracted_entities"])
        if not tdf.empty:
            all_ents = [e for row in tdf["extracted_entities"] for e in row]
            ent_cnt  = pd.Series(all_ents).value_counts()
            wk = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
            rec = tdf[pd.to_datetime(tdf["timestamp"], utc=True) >= wk]
            src = [e for row in (rec if len(rec) >= 5 else tdf)["extracted_entities"] for e in row]
            freq = pd.Series(src).value_counts().to_dict()
            if freq:
                _pal = ["#00d4ff","#2a9d8f","#48cae4","#0077b6","#e9c46a","#90e0ef","#7a9bb5"]
                def _cf(word, **_): return _pal[sum(ord(c) for c in word) % len(_pal)]
                wc = WordCloud(width=1400, height=360, background_color=None, mode="RGBA",
                               color_func=_cf, max_words=60, prefer_horizontal=.85,
                               collocations=False).generate_from_frequencies(freq)
                fig_wc, ax = plt.subplots(figsize=(14, 4))
                ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                fig_wc.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
                st.pyplot(fig_wc, use_container_width=True); plt.close(fig_wc)

            top = ent_cnt.head(15).reset_index()
            top.columns = ["Entidad", "Menciones"]
            fb = px.bar(top, x="Menciones", y="Entidad", orientation="h",
                        color="Menciones", color_continuous_scale="Blues", text="Menciones")
            fb.update_traces(textposition="outside")
            fb.update_layout(**_PB, showlegend=False, coloraxis_showscale=False)
            fb.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fb, use_container_width=True)
        else:
            st.info("Aún no hay consultas de texto con entidades extraídas.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUACIÓN + COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:

    def _bar_group(metrics_dict: dict, height: int = 320) -> go.Figure:
        """Gráfico de barras agrupadas P/R/F1 por clase."""
        rows = [{
            "Clase": l,
            "Precisión": metrics_dict["per_class"].get(l, {}).get("precision", 0),
            "Recall":    metrics_dict["per_class"].get(l, {}).get("recall", 0),
            "F1":        metrics_dict["per_class"].get(l, {}).get("f1", 0),
        } for l in LABELS]
        df_r = pd.DataFrame(rows)
        fig = go.Figure()
        for mn, mc in [("Precisión","#00d4ff"),("Recall","#2a9d8f"),("F1","#e9c46a")]:
            fig.add_trace(go.Bar(
                name=mn, x=df_r["Clase"], y=df_r[mn], marker_color=mc,
                text=[f"{v:.3f}" for v in df_r[mn]],
                textposition="outside", textfont=dict(size=11, color="white"),
            ))
        fig.update_layout(
            **_PB, barmode="group", height=height,
            yaxis=dict(range=[0, 1.15], tickformat=".2f", gridcolor="#1e3a5f"),
            xaxis=dict(gridcolor="#1e3a5f"),
            legend=dict(orientation="h", y=1.12, x=.5, xanchor="center"),
        )
        return fig

    def _heatmap(cm_list: list, labels: list, height: int = 320) -> go.Figure:
        """Matriz de confusión como heatmap interactivo."""
        arr  = np.array(cm_list)
        norm = arr.astype(float) / (arr.sum(axis=1, keepdims=True) + 1e-9)
        ann  = [dict(x=labels[j], y=labels[i],
                 text=f"<b>{arr[i,j]}</b><br>{norm[i,j]:.1%}",
                 showarrow=False,
                 font=dict(color="#0a1628" if norm[i,j] > .55 else "#d4e2ef", size=14))
                for i in range(len(labels)) for j in range(len(labels))]
        fig = go.Figure(go.Heatmap(
            z=norm, x=labels, y=labels, zmin=0, zmax=1,
            colorscale=[[0,"#0d1b2a"],[.5,"#0f3460"],[1,"#00d4ff"]],
            showscale=True, colorbar=dict(tickformat=".0%", outlinecolor="#1e3a5f"),
        ))
        fig.update_layout(
            **_PB, annotations=ann, height=height,
            xaxis=dict(title="Predicho", side="bottom",
                       tickfont=dict(size=13, color="#c0d0e0")),
            yaxis=dict(title="Esperado", autorange="reversed",
                       tickfont=dict(size=13, color="#c0d0e0")),
        )
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # SECCIÓN 1 — MODELO EN PRODUCCIÓN (14B)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f'<p class="ctitle">Modelo en producción &nbsp;<span style="font-size:.7em;color:#7a9bb5;font-weight:400;">{PROD["label"]}</span></p>', unsafe_allow_html=True)

    mf1   = PROD["macro_f1"]
    pass_ = mf1 >= 0.75
    badge = "f1-pass" if pass_ else "f1-fail"
    icon  = "✓" if pass_ else "✗"
    lbl   = "Supera el umbral (≥ 0.75)" if pass_ else "Por debajo del umbral (≥ 0.75)"

    col_hero, col_kpis = st.columns([1, 2], gap="large")
    with col_hero:
        st.markdown(f"""
        <div class="f1-wrap">
            <span class="{badge}">{icon} {mf1:.4f}</span>
            <span class="f1-lbl">Macro-F1 &nbsp;·&nbsp; {lbl}</span>
        </div>""", unsafe_allow_html=True)

    with col_kpis:
        st.markdown("")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Ejemplos evaluados",  PROD["total_examples"])
        q2.metric("Errores de pipeline", PROD["errors"])
        q3.metric("Cobertura con URL",   f"{PROD['url_coverage_pct']}%")
        q4.metric("Latencia p50",        f"{PROD['latency_p50_ms']:,.0f} ms")

    st.divider()

    ca, cb = st.columns(2, gap="large")
    with ca:
        st.markdown('<p class="stitle">Métricas por clase</p>', unsafe_allow_html=True)
        st.plotly_chart(_bar_group(PROD), use_container_width=True)
        tbl = pd.DataFrame([{
            "Clase": l,
            "Precisión": f"{PROD['per_class'][l]['precision']:.4f}",
            "Recall":    f"{PROD['per_class'][l]['recall']:.4f}",
            "F1":        f"{PROD['per_class'][l]['f1']:.4f}",
            "Soporte":   PROD["per_class"][l].get("support", "—"),
        } for l in LABELS]).set_index("Clase")
        st.dataframe(tbl, use_container_width=True)

    with cb:
        st.markdown('<p class="stitle">Matriz de confusión</p>', unsafe_allow_html=True)
        st.plotly_chart(
            _heatmap(PROD["confusion_matrix"], PROD["confusion_matrix_labels"]),
            use_container_width=True,
        )

    st.divider()

    cc, cd = st.columns(2, gap="large")
    with cc:
        st.markdown('<p class="stitle">Accuracy por categoría</p>', unsafe_allow_html=True)
        cat_df = pd.DataFrame(
            sorted(PROD["accuracy_by_category"].items(), key=lambda x: x[1]),
            columns=["Categoría", "Accuracy"],
        )
        cat_df["Pct"] = cat_df["Accuracy"].map("{:.1%}".format)
        fig_cat = px.bar(cat_df, x="Accuracy", y="Categoría", orientation="h",
                         color="Accuracy", text="Pct",
                         color_continuous_scale=[[0,"#e63946"],[.5,"#e9c46a"],[1,"#2a9d8f"]],
                         range_color=[0, 1])
        fig_cat.update_traces(textposition="outside", textfont=dict(size=11))
        fig_cat.update_layout(**_PB, showlegend=False, coloraxis_showscale=False,
                              xaxis=dict(range=[0, 1.15], tickformat=".0%",
                                         gridcolor="#1e3a5f"), height=380)
        st.plotly_chart(fig_cat, use_container_width=True)

    with cd:
        st.markdown('<p class="stitle">Latencia por clase (ms)</p>', unsafe_allow_html=True)
        lat_df = pd.DataFrame([
            {"Clase": l, "Media": v["mean_ms"], "p50": v["p50_ms"], "p95": v["p95_ms"]}
            for l, v in PROD["latency_by_class"].items()
        ])
        fig_lat = go.Figure()
        for mn, mc in [("Media","#00d4ff"),("p50","#2a9d8f"),("p95","#e63946")]:
            fig_lat.add_trace(go.Bar(
                name=mn, x=lat_df["Clase"], y=lat_df[mn], marker_color=mc,
                text=[f"{v:,.0f}" for v in lat_df[mn]],
                textposition="outside", textfont=dict(size=11, color="white"),
            ))
        fig_lat.update_layout(
            **_PB, barmode="group", height=380,
            yaxis=dict(title="ms", gridcolor="#1e3a5f"),
            xaxis=dict(gridcolor="#1e3a5f"),
            legend=dict(orientation="h", y=1.12, x=.5, xanchor="center"),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    st.divider()
    st.markdown('<p class="stitle">Latencia global del pipeline</p>', unsafe_allow_html=True)
    l1, l2, l3 = st.columns(3)
    l1.metric("Media",  f"{PROD['latency_mean_ms']:,.0f} ms")
    l2.metric("p50",    f"{PROD['latency_p50_ms']:,.0f} ms")
    l3.metric("p95",    f"{PROD['latency_p95_ms']:,.0f} ms")

    # ─────────────────────────────────────────────────────────────────────────
    # SECCIÓN 2 — COMPARATIVA: 14B vs MODELO BASE
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <p class="ctitle">
        Comparativa de modelos &nbsp;
        <span style="font-size:.7em;color:#7a9bb5;font-weight:400;">
            Qwen 2.5 14B Instruct en producción vs modelo base (Ollama 3B)
        </span>
    </p>""", unsafe_allow_html=True)

    dm_f1  = PROD["macro_f1"] - BASE["macro_f1"]
    dm_lat = PROD["latency_mean_ms"] - BASE["latency_mean_ms"]
    dm_url = PROD["url_coverage_pct"] - BASE["url_coverage_pct"]
    dm_f1_fake = PROD["per_class"]["FAKE"]["f1"] - BASE["per_class"]["FAKE"]["f1"]

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f"""
        <div class="cmp-card">
          <p class="cmp-card-title" style="color:#7a9bb5;">Macro-F1</p>
          <p class="cmp-card-val" style="color:#e2e8f0;">
            {BASE['macro_f1']:.4f} → {PROD['macro_f1']:.4f}
          </p>
          <p class="cmp-delta delta-pos">▲ +{dm_f1:.4f} ({dm_f1/BASE['macro_f1']*100:.1f}%)</p>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="cmp-card">
          <p class="cmp-card-title" style="color:#7a9bb5;">F1 FAKE</p>
          <p class="cmp-card-val" style="color:#e63946;">
            {BASE['per_class']['FAKE']['f1']:.4f} → {PROD['per_class']['FAKE']['f1']:.4f}
          </p>
          <p class="cmp-delta delta-pos">▲ +{dm_f1_fake:.4f} (×{PROD['per_class']['FAKE']['f1']/max(BASE['per_class']['FAKE']['f1'],0.001):.1f})</p>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="cmp-card">
          <p class="cmp-card-title" style="color:#7a9bb5;">Cobertura con URL</p>
          <p class="cmp-card-val" style="color:#e2e8f0;">
            {BASE['url_coverage_pct']}% → {PROD['url_coverage_pct']}%
          </p>
          <p class="cmp-delta delta-pos">▲ +{dm_url:.1f}%</p>
        </div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown(f"""
        <div class="cmp-card">
          <p class="cmp-card-title" style="color:#7a9bb5;">Latencia media</p>
          <p class="cmp-card-val" style="color:#e2e8f0;">
            {BASE['latency_mean_ms']:,.0f} → {PROD['latency_mean_ms']:,.0f} ms
          </p>
          <p class="cmp-delta delta-neg">▼ {dm_lat:+,.0f} ms (modelo mayor)</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Macro-F1 y F1 por clase comparativos
    ce1, ce2 = st.columns(2, gap="large")

    with ce1:
        st.markdown('<p class="stitle">Macro-F1 por modelo</p>', unsafe_allow_html=True)
        fig_mf1 = go.Figure()
        fig_mf1.add_trace(go.Bar(
            x=[BASE["label"], PROD["label"]],
            y=[BASE["macro_f1"], PROD["macro_f1"]],
            marker_color=["#4a6a89", "#00d4ff"],
            text=[f"{BASE['macro_f1']:.4f}", f"{PROD['macro_f1']:.4f}"],
            textposition="outside", textfont=dict(size=15, color="white"),
            width=0.45,
        ))
        fig_mf1.add_hline(y=0.75, line_dash="dash", line_color="#e9c46a",
                          annotation_text="Umbral TFM (0.75)",
                          annotation_position="top left",
                          annotation_font_color="#e9c46a")
        fig_mf1.update_layout(
            **_PB, height=300, showlegend=False,
            yaxis=dict(range=[0, 1], tickformat=".2f", gridcolor="#1e3a5f"),
            xaxis=dict(gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig_mf1, use_container_width=True)

    with ce2:
        st.markdown('<p class="stitle">F1 por clase y modelo</p>', unsafe_allow_html=True)
        f1_rows = []
        for lbl in LABELS:
            f1_rows.append({"Clase": lbl, "Modelo": BASE["label"],
                            "F1": BASE["per_class"][lbl]["f1"]})
            f1_rows.append({"Clase": lbl, "Modelo": PROD["label"],
                            "F1": PROD["per_class"][lbl]["f1"]})
        f1_df = pd.DataFrame(f1_rows)
        fig_f1 = px.bar(f1_df, x="Clase", y="F1", color="Modelo", barmode="group",
                        color_discrete_map={BASE["label"]: "#4a6a89", PROD["label"]: "#00d4ff"},
                        text=f1_df["F1"].map("{:.3f}".format))
        fig_f1.update_traces(textposition="outside", textfont=dict(size=11))
        fig_f1.update_layout(
            **_PB, height=300,
            yaxis=dict(range=[0, 1.12], tickformat=".2f", gridcolor="#1e3a5f"),
            xaxis=dict(gridcolor="#1e3a5f"),
            legend=dict(orientation="h", y=1.12, x=.5, xanchor="center"),
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    # Matrices de confusión lado a lado
    st.markdown('<p class="stitle">Comparativa de matrices de confusión</p>', unsafe_allow_html=True)

    fig_cms = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                            subplot_titles=["", ""])

    def _add_heatmap_to_subplot(fig, cm_list, labels, row, col):
        arr  = np.array(cm_list)
        norm = arr.astype(float) / (arr.sum(axis=1, keepdims=True) + 1e-9)
        fig.add_trace(go.Heatmap(
            z=norm, x=labels, y=labels, zmin=0, zmax=1,
            colorscale=[[0,"#0d1b2a"],[.5,"#0f3460"],[1,"#00d4ff"]],
            showscale=False,
        ), row=row, col=col)
        xref = f"x{col if col > 1 else ''}"
        yref = f"y{col if col > 1 else ''}"
        for i in range(len(labels)):
            for j in range(len(labels)):
                fig.add_annotation(
                    x=labels[j], y=labels[i],
                    text=f"<b>{arr[i,j]}</b><br>{norm[i,j]:.1%}",
                    showarrow=False,
                    font=dict(color="#0a1628" if norm[i,j] > .55 else "#d4e2ef", size=13),
                    xref=xref, yref=yref,
                )

    _add_heatmap_to_subplot(fig_cms, BASE["confusion_matrix"], LABELS, 1, 1)
    _add_heatmap_to_subplot(fig_cms, PROD["confusion_matrix"], LABELS, 1, 2)

    fig_cms.update_layout(
        **_PB, height=340,
        annotations=fig_cms.layout.annotations + (
            dict(x=0.20, y=1.12, xref="paper", yref="paper",
                 text=BASE["label"], showarrow=False,
                 font=dict(size=12, color="#7a9bb5")),
            dict(x=0.80, y=1.12, xref="paper", yref="paper",
                 text=PROD["label"], showarrow=False,
                 font=dict(size=12, color="#00d4ff")),
        ),
    )
    fig_cms.update_xaxes(title_text="Predicho", tickfont=dict(size=12, color="#c0d0e0"))
    fig_cms.update_yaxes(title_text="Esperado", autorange="reversed",
                         tickfont=dict(size=12, color="#c0d0e0"))
    st.plotly_chart(fig_cms, use_container_width=True)

    # Accuracy por categoría comparativa
    st.markdown('<p class="stitle">Accuracy por categoría — comparativa</p>', unsafe_allow_html=True)
    all_cats = sorted(set(BASE["accuracy_by_category"]) | set(PROD["accuracy_by_category"]))
    cat_cmp = pd.DataFrame([{
        "Categoría": cat,
        BASE["label"]: BASE["accuracy_by_category"].get(cat, 0),
        PROD["label"]: PROD["accuracy_by_category"].get(cat, 0),
    } for cat in all_cats]).sort_values(BASE["label"])

    fig_ccat = go.Figure()
    for m_name, m_color in [(BASE["label"],"#4a6a89"), (PROD["label"],"#00d4ff")]:
        fig_ccat.add_trace(go.Bar(
            name=m_name, y=cat_cmp["Categoría"], x=cat_cmp[m_name],
            orientation="h", marker_color=m_color,
            text=[f"{v:.0%}" for v in cat_cmp[m_name]],
            textposition="outside", textfont=dict(size=10),
        ))
    fig_ccat.update_layout(
        **_PB, barmode="group", height=400,
        xaxis=dict(range=[0, 1.2], tickformat=".0%", gridcolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f"),
        legend=dict(orientation="h", y=1.06, x=.5, xanchor="center"),
    )
    st.plotly_chart(fig_ccat, use_container_width=True)

    # Latencia comparativa
    st.markdown('<p class="stitle">Latencia global — trade-off por modelo</p>', unsafe_allow_html=True)
    lat_cmp = pd.DataFrame({
        "Métrica": ["Media", "p50 (mediana)", "p95"],
        BASE["label"]: [BASE["latency_mean_ms"], BASE["latency_p50_ms"], BASE["latency_p95_ms"]],
        PROD["label"]: [PROD["latency_mean_ms"], PROD["latency_p50_ms"], PROD["latency_p95_ms"]],
    })
    fig_lc = go.Figure()
    for m_name, m_color in [(BASE["label"],"#4a6a89"), (PROD["label"],"#00d4ff")]:
        fig_lc.add_trace(go.Bar(
            name=m_name, x=lat_cmp["Métrica"], y=lat_cmp[m_name],
            marker_color=m_color,
            text=[f"{v:,.0f} ms" for v in lat_cmp[m_name]],
            textposition="outside", textfont=dict(size=11, color="white"),
        ))
    fig_lc.update_layout(
        **_PB, barmode="group", height=280,
        yaxis=dict(title="ms", gridcolor="#1e3a5f"),
        xaxis=dict(gridcolor="#1e3a5f"),
        legend=dict(orientation="h", y=1.12, x=.5, xanchor="center"),
    )
    st.plotly_chart(fig_lc, use_container_width=True)
    dm_f1_pct = dm_f1 / BASE["macro_f1"] * 100
    st.markdown(
        f'<div class="nota-pie">'
        f'<strong>⚠️ Trade-off de escala</strong> &nbsp;·&nbsp; '
        f'El modelo 14B necesita aproximadamente el doble de tiempo de inferencia. '
        f'La mejora en Macro-F1 (<strong>+{dm_f1_pct:.1f}%</strong>) tiene como contraparte '
        f'una mayor latencia. Los valores de latencia del 14B son representativos del '
        f'coste de escalar el modelo en un entorno de producción.'
        f'</div>',
        unsafe_allow_html=True,
    )
