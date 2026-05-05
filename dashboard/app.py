"""
VERUM Analytics Dashboard — Streamlit app.
Connects to MongoDB and displays real-time usage metrics.

Run locally: streamlit run dashboard/app.py
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

st.set_page_config(page_title="VERUM Dashboard", layout="wide")
st.title("VERUM — Intelligence Dashboard")


@st.cache_resource
def get_collection():
    uri = (
        f"mongodb://{os.environ['MONGO_USER']}:{os.environ['MONGO_PASS']}"
        f"@{os.environ.get('MONGO_HOST', 'localhost')}:{os.environ.get('MONGO_PORT', 27017)}"
    )
    client = MongoClient(uri)
    return client[os.environ["MONGO_DB"]][os.environ.get("MONGO_COLLECTION_QUERIES", "queries")]


@st.cache_data(ttl=30)
def load_data() -> pd.DataFrame:
    col = get_collection()
    docs = list(col.find({}, {"_id": 0}).sort("timestamp", -1).limit(1000))
    return pd.DataFrame(docs) if docs else pd.DataFrame()


df = load_data()

if df.empty:
    st.info("No data yet. Start interacting with the bot.")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total queries",    len(df))
col2.metric("FAKE detected",    (df["final_verdict"] == "FAKE").sum())
col3.metric("Avg latency (ms)", int(df["total_processing_time_ms"].mean()))
col4.metric("Unique users",     df["user_hash"].nunique())

st.divider()

# ── Verdict distribution ──────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.subheader("Verdict distribution")
    fig = px.pie(df, names="final_verdict", color="final_verdict",
                 color_discrete_map={"FAKE": "#e63946", "REAL": "#2a9d8f", "UNVERIFIED": "#e9c46a"})
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Queries over time")
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    timeline = df.groupby("date").size().reset_index(name="count")
    st.plotly_chart(px.line(timeline, x="date", y="count"), use_container_width=True)

# ── Entity word cloud (text queries) ─────────────────────────────────────────
st.subheader("Top extracted entities")
text_df = df[df["payload_type"] == "text"].dropna(subset=["extracted_entities"])
if not text_df.empty:
    all_entities = [e for row in text_df["extracted_entities"] for e in row]
    entity_counts = pd.Series(all_entities).value_counts().head(20).reset_index()
    entity_counts.columns = ["entity", "count"]
    st.plotly_chart(px.bar(entity_counts, x="entity", y="count", color="count"), use_container_width=True)
