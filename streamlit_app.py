import math
import re
import tempfile
from itertools import combinations

import boto3
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# =========================================================
# Page / Style
# =========================================================

st.set_page_config(page_title="IT 뉴스 분석 대시보드", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a 0%,#111827 45%,#020617 100%); color:#e5e7eb;}
.block-container {padding-top:1.6rem; padding-bottom:3rem;}
h1 {color:#f8fafc; font-weight:900; letter-spacing:-0.05em;}
h2,h3 {color:#e5e7eb; font-weight:850;}
[data-testid="stCaptionContainer"] {color:#94a3b8;}
[data-testid="stDataFrame"] {border-radius:14px; overflow:hidden; border:1px solid rgba(148,163,184,.18);}
.stAlert {background:rgba(14,165,233,.12); border:1px solid rgba(56,189,248,.35); border-radius:16px; color:#e0f2fe;}
hr {border:none; height:1px; background:linear-gradient(90deg,transparent,#38bdf8,transparent); margin:2rem 0;}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15,23,42,.62);
    border: 1px solid rgba(148,163,184,.16);
    padding: 10px;
    border-radius: 18px;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 13px;
    padding: 0 14px;
    background: rgba(2,6,23,.45);
    border: 1px solid rgba(148,163,184,.10);
    color: #94a3b8;
    font-weight: 800;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(14,165,233,.95), rgba(37,99,235,.92)) !important;
    color: #ffffff !important;
    border: 1px solid rgba(125,211,252,.8) !important;
}
.stTabs [data-baseweb="tab-highlight"] {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("IT 뉴스 데이터 분석 대시보드")
st.caption("뉴스 수집 결과를 단순 나열하지 않고, 데이터 품질 검증 → 분석기법 적용 → 결과 해석 → 한계점 확인 순서로 구성했습니다.")

# =========================================================
# Config
# =========================================================

AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = st.secrets["AWS_REGION"]
BUCKET_NAME = st.secrets["BUCKET_NAME"]

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

S3_PREFIX = "it_news/IT/processed/"
START_DATE = "20260514"

CORE_KEYWORDS = [
    "AI", "인공지능", "생성형AI", "챗GPT", "ChatGPT", "GPT", "OpenAI", "LLM",
    "반도체", "AI반도체", "AI 반도체", "HBM", "GPU", "엔비디아", "NVIDIA",
    "삼성전자", "SK하이닉스", "파운드리", "메모리",
    "클라우드", "AWS", "Azure", "데이터센터", "데이터 센터", "서버", "인프라",
    "보안", "해킹", "개인정보", "랜섬웨어", "침해", "취약점", "사이버",
    "로봇", "전기차", "배터리", "자율주행", "모빌리티",
    "네이버", "카카오", "구글", "애플", "메타", "마이크로소프트", "MS",
    "데이터", "빅데이터", "소프트웨어", "디지털", "DX", "핀테크", "블록체인"
]

TOPIC_MAP = {
    "AI/인공지능": ["AI", "인공지능", "생성형AI", "챗GPT", "ChatGPT", "GPT", "OpenAI", "LLM", "엔비디아", "GPU"],
    "반도체": ["반도체", "AI반도체", "AI 반도체", "HBM", "삼성전자", "SK하이닉스", "파운드리", "메모리", "칩"],
    "클라우드/인프라": ["클라우드", "AWS", "Azure", "데이터센터", "데이터 센터", "서버", "인프라", "SaaS", "쿠버네티스"],
    "보안/리스크": ["보안", "해킹", "개인정보", "랜섬웨어", "침해", "취약점", "사이버", "정보보호"],
    "모빌리티/로봇": ["전기차", "자율주행", "배터리", "로봇", "모빌리티", "이차전지"],
    "플랫폼/빅테크": ["네이버", "카카오", "구글", "애플", "메타", "마이크로소프트", "MS", "플랫폼", "빅테크"],
}

EVENT_KEYWORDS = {
    "월드IT쇼": ["월드IT쇼", "World IT Show", "WIS"],
    "CES": ["CES"],
    "MWC": ["MWC"],
    "전시/컨퍼런스": ["컨퍼런스", "박람회", "전시회", "행사", "세미나", "포럼"]
}

POSITIVE_WORDS = ["성장", "확대", "출시", "투자", "협력", "개선", "강화", "수주", "증가", "성공", "최초", "고도화", "혁신", "개발"]
RISK_WORDS = ["해킹", "침해", "유출", "장애", "중단", "규제", "감소", "적자", "위험", "논란", "피해", "취약점", "공격", "랜섬웨어"]

BAD_TOKENS = ["nbsp", "quot", "amp", "lt", "gt", "may", "assumed", "errors", "said", "the", "and", "for", "to", "in", "is"]
STOPWORDS = set([
    "the", "and", "for", "that", "with", "this", "from", "have", "will", "into", "about",
    "their", "they", "them", "were", "been", "being", "said", "more", "than", "over",
    "after", "before", "while", "where", "when", "what", "which", "would", "could", "should",
    "there", "these", "those", "because", "through", "during", "under", "between", "among",
    "it", "to", "of", "in", "is", "on", "at", "by", "be", "as", "an", "or", "if", "we",
    "he", "she", "you", "are", "was", "has", "had", "can", "not", "also", "but", "how",
    "뉴스", "기사", "기자", "보도", "관련", "대한", "위해", "통해", "이번", "최근", "오늘", "밝혔다", "말했다"
])

# =========================================================
# UI Helpers
# =========================================================

def section(title, subtitle=None):
    st.markdown("---")
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)

def card(title, value, desc="", color="#38bdf8"):
    st.markdown(f"""
    <div style="background:rgba(15,23,42,.82);border:1px solid rgba(56,189,248,.25);
    border-radius:20px;padding:20px;min-height:138px;box-shadow:0 0 20px rgba(56,189,248,.06);">
      <div style="color:#94a3b8;font-size:13px;font-weight:750;margin-bottom:8px;">{title}</div>
      <div style="color:{color};font-size:28px;font-weight:950;margin-bottom:8px;line-height:1.2;">{value}</div>
      <div style="color:#cbd5e1;font-size:13px;line-height:1.55;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

def method_box(method, purpose, formula, interpretation):
    st.markdown(f"""
    <div style="background:rgba(2,6,23,.72);border:1px solid rgba(148,163,184,.18);
    border-radius:18px;padding:18px;margin-bottom:12px;">
      <div style="font-size:18px;font-weight:900;color:#f8fafc;margin-bottom:10px;">분석기법: {method}</div>
      <div style="color:#cbd5e1;line-height:1.7;"><b>분석 목적</b>: {purpose}</div>
      <div style="color:#cbd5e1;line-height:1.7;"><b>계산 방식</b>: {formula}</div>
      <div style="color:#e0f2fe;line-height:1.7;"><b>해석 기준</b>: {interpretation}</div>
    </div>
    """, unsafe_allow_html=True)

def progress_list(df, label_col, value_col, title=None, top_n=10, suffix="건"):
    if title:
        st.markdown(f"### {title}")
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        st.warning("표시할 데이터가 없습니다.")
        return

    view = df.head(top_n).copy()
    max_v = view[value_col].max()
    for _, row in view.iterrows():
        label, value = row[label_col], float(row[value_col])
        ratio = value / max_v * 100 if max_v else 0
        value_text = f"{int(value):,}{suffix}" if value == int(value) else f"{value:.2f}{suffix}"
        st.markdown(f"""
        <div style="background:rgba(15,23,42,.75);border:1px solid rgba(148,163,184,.12);border-radius:16px;
        padding:13px 16px;margin-bottom:9px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">
            <div style="font-size:15px;font-weight:850;color:white;">{label}</div>
            <div style="font-size:14px;font-weight:850;color:#38bdf8;">{value_text}</div>
          </div>
          <div style="width:100%;height:8px;background:#1e293b;border-radius:999px;overflow:hidden;">
            <div style="width:{ratio}%;height:100%;background:linear-gradient(90deg,#38bdf8,#60a5fa);border-radius:999px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

def insight_box(title, bullets):
    body = "".join([f"<li>{b}</li>" for b in bullets])
    st.markdown(f"""
    <div style="background:rgba(14,165,233,.12);border:1px solid rgba(56,189,248,.35);
    border-radius:18px;padding:18px;margin:12px 0;">
      <div style="font-size:18px;font-weight:950;color:#e0f2fe;margin-bottom:8px;">{title}</div>
      <ul style="color:#dbeafe;line-height:1.75;margin-bottom:0;">{body}</ul>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# S3 Loading
# =========================================================

@st.cache_data(ttl=600)
def list_csv_keys(bucket, prefix):
    files, token = [], None
    while True:
        params = {"Bucket": bucket, "Prefix": prefix}
        if token:
            params["ContinuationToken"] = token
        res = s3.list_objects_v2(**params)
        files.extend(res.get("Contents", []))
        if not res.get("IsTruncated"):
            break
        token = res.get("NextContinuationToken")
    return [f["Key"] for f in files if f["Key"].endswith(".csv")]

def key_date(key):
    m = re.search(r"(\d{8})", key)
    return m.group(1) if m else None

@st.cache_data(ttl=600)
def load_csvs(bucket, keys):
    dfs = []
    for key in keys:
        obj = s3.get_object(Bucket=bucket, Key=key)
        tmp = pd.read_csv(obj["Body"])
        tmp["loaded_file"] = key
        dfs.append(tmp)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def normalize_date(df):
    if "analysis_date" in df.columns:
        df["analysis_date"] = pd.to_datetime(df["analysis_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("unknown")
        return df, "analysis_date"
    date_col = next((c for c in ["pubDate_dt", "pubDate_ymd", "pubDate", "date", "published_date"] if c in df.columns), None)
    if not date_col:
        df["analysis_date"] = "unknown"
        return df, "analysis_date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["analysis_date"] = df[date_col].dt.strftime("%Y-%m-%d").fillna("unknown")
    return df, "analysis_date"

def prepare_df(df):
    df, date_col = normalize_date(df)
    for col in ["source", "source_group", "title", "description", "originallink", "link", "event_tag"]:
        if col not in df.columns:
            df[col] = ""

    for col in ["source", "source_group", "title", "description", "originallink", "link", "event_tag"]:
        df[col] = df[col].fillna("").astype(str)

    df.loc[df["source"].str.strip() == "", "source"] = "unknown"
    df.loc[df["source_group"].str.strip() == "", "source_group"] = df["source"]
    df["analysis_source"] = df["source"]

    dedup_cols = [c for c in ["originallink", "link", "title"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last")
    return df, date_col

# =========================================================
# Analysis Functions
# =========================================================

def text_series(df):
    return df["title"].fillna("").astype(str) + " " + df["description"].fillna("").astype(str)

def contains_any(text, keywords):
    text = str(text)
    return any(str(kw).lower() in text.lower() for kw in keywords)

def filter_keyword(df, keyword):
    return df[text_series(df).str.contains(str(keyword), case=False, regex=False)]

def filter_keywords(df, keywords):
    txt = text_series(df)
    cond = pd.Series(False, index=df.index)
    for kw in keywords:
        cond |= txt.str.contains(str(kw), case=False, regex=False)
    return df[cond]

def keyword_counts(df, keywords):
    txt = text_series(df)
    rows = []
    for kw in keywords:
        cnt = int(txt.str.contains(str(kw), case=False, regex=False).sum())
        if cnt > 0:
            rows.append({"keyword": kw, "count": cnt})
    return pd.DataFrame(rows).sort_values("count", ascending=False) if rows else pd.DataFrame(columns=["keyword", "count"])

def tfidf_keyword_scores(df, candidates=CORE_KEYWORDS, top_n=20):
    txt = text_series(df)
    total_docs = len(txt)
    rows = []
    for kw in candidates:
        contains = txt.str.contains(str(kw), case=False, regex=False)
        article_count = int(contains.sum())
        if article_count == 0:
            continue
        idf = 1 + math.log((1 + total_docs) / (1 + article_count))
        score = article_count * idf
        rows.append({
            "keyword": kw,
            "article_count": article_count,
            "idf": round(idf, 4),
            "tfidf_score": round(score, 4),
            "coverage_ratio": round(article_count / total_docs * 100, 2) if total_docs else 0
        })
    return pd.DataFrame(rows).sort_values("tfidf_score", ascending=False).head(top_n) if rows else pd.DataFrame()

def classify_topic_for_row(text):
    matched = []
    for topic, kws in TOPIC_MAP.items():
        if contains_any(text, kws):
            matched.append(topic)
    return matched[0] if matched else "기타"

def add_topic_columns(df):
    out = df.copy()
    txt = text_series(out)
    out["primary_topic"] = txt.apply(classify_topic_for_row)
    return out

def classify_sentiment_text(text):
    pos = sum(w in str(text) for w in POSITIVE_WORDS)
    risk = sum(w in str(text) for w in RISK_WORDS)
    if risk > pos:
        return "부정/리스크"
    if pos > risk:
        return "긍정/성장"
    return "중립"

def add_sentiment(df):
    out = df.copy()
    out["sentiment_group"] = text_series(out).apply(classify_sentiment_text)
    return out

def detect_event_text(text):
    tags = []
    for tag, kws in EVENT_KEYWORDS.items():
        if contains_any(text, kws):
            tags.append(tag)
    return ", ".join(tags)

def add_event_tags(df):
    out = df.copy()
    calculated = text_series(out).apply(detect_event_text)
    existing = out["event_tag"].fillna("").astype(str) if "event_tag" in out.columns else ""
    out["event_detected"] = existing.where(existing.str.strip() != "", calculated)
    out["is_event_article"] = out["event_detected"].str.strip() != ""
    return out

def daily_counts(df, date_col):
    return df.groupby(date_col).size().reset_index(name="article_count").sort_values(date_col)

def daily_keyword_matrix(df, date_col, keywords):
    rows = []
    for date in sorted(df[date_col].dropna().unique()):
        sub = df[df[date_col] == date]
        counts = keyword_counts(sub, keywords)
        for _, row in counts.iterrows():
            rows.append({"date": date, "keyword": row["keyword"], "count": int(row["count"])})
    return pd.DataFrame(rows)

def surge_sustain_analysis(df, date_col, keywords):
    daily = daily_keyword_matrix(df, date_col, keywords)
    if daily.empty:
        return pd.DataFrame()

    pivot = daily.pivot_table(index="date", columns="keyword", values="count", aggfunc="sum", fill_value=0).sort_index()
    rows = []
    for kw in pivot.columns:
        s = pivot[kw].astype(float)
        total_count = int(s.sum())
        active_days = int((s > 0).sum())
        max_count = int(s.max())
        max_date = s.idxmax()
        prev = s.shift(1)
        growth = []
        for today, yesterday in zip(s.iloc[1:], prev.iloc[1:]):
            if yesterday == 0 and today > 0:
                growth.append(100.0)
            elif yesterday > 0:
                growth.append((today - yesterday) / yesterday * 100)
        max_growth_rate = round(max(growth), 1) if growth else 0.0

        total_days = len(s)
        if active_days >= max(3, int(total_days * 0.6)):
            trend_type = "지속형"
        elif max_growth_rate >= 100:
            trend_type = "급등형"
        elif active_days <= 2 and max_count >= 5:
            trend_type = "이벤트형"
        else:
            trend_type = "관찰형"

        rows.append({
            "keyword": kw,
            "total_count": total_count,
            "active_days": active_days,
            "max_count": max_count,
            "max_date": max_date,
            "max_growth_rate": max_growth_rate,
            "trend_type": trend_type
        })
    return pd.DataFrame(rows).sort_values(["trend_type", "total_count"], ascending=[True, False])

def topic_summary(df):
    counts = df["primary_topic"].value_counts().reset_index()
    counts.columns = ["topic", "article_count"]
    total = len(df)
    counts["ratio"] = (counts["article_count"] / total * 100).round(1) if total else 0
    return counts

def topic_timeseries(df, date_col):
    pivot = df.pivot_table(index=date_col, columns="primary_topic", values="title", aggfunc="count", fill_value=0).reset_index()
    return pivot.rename(columns={date_col: "date"})

def keyword_network(df, keywords):
    rows = []
    for text in text_series(df):
        appeared = sorted({kw for kw in keywords if str(kw).lower() in str(text).lower()})
        rows.extend(combinations(appeared, 2))
    if not rows:
        return pd.DataFrame(columns=["keyword_a", "keyword_b", "co_count"])
    return pd.DataFrame(rows, columns=["keyword_a", "keyword_b"]).value_counts().reset_index(name="co_count").sort_values("co_count", ascending=False)

def network_html(net_df, top_n=25):
    G = nx.Graph()
    for _, row in net_df.head(top_n).iterrows():
        weight = int(row["co_count"])
        G.add_edge(row["keyword_a"], row["keyword_b"], value=max(1, weight), title=f"{row['keyword_a']} ↔ {row['keyword_b']} | {weight:,}건")

    net = Network(height="680px", width="100%", bgcolor="#0f172a", font_color="white", notebook=False)
    net.from_nx(G)

    degrees = dict(G.degree())
    weighted = dict(G.degree(weight="value"))
    for node in net.nodes:
        node_id = node["id"]
        node["size"] = min(76, max(24, 20 + weighted.get(node_id, 1) * 0.55))
        node["color"] = {"background": "#38bdf8", "border": "#7dd3fc"}
        node["font"] = {"size": 18, "face": "Arial", "color": "#f8fafc", "strokeWidth": 3, "strokeColor": "#0f172a"}
        node["title"] = f"{node_id}<br>연결 키워드 수: {degrees.get(node_id, 0)}"

    for edge in net.edges:
        edge["color"] = {"color": "rgba(148,163,184,.72)", "highlight": "#38bdf8"}
        edge["width"] = min(10, max(1, edge.get("value", 1) / 5))

    net.set_options("""
    var options = {
      "interaction": {"hover": true, "dragNodes": true, "dragView": true, "zoomView": true},
      "nodes": {"shape": "dot"},
      "physics": {"enabled": false}
    }
    """)
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    net.save_graph(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def data_quality_checks(df, date_col):
    txt = text_series(df).str.lower()
    bad_rows = []
    for token in BAD_TOKENS:
        count = int(txt.str.contains(token.lower(), regex=False).sum())
        if count > 0:
            bad_rows.append({"token": token, "article_count": count})
    bad_df = pd.DataFrame(bad_rows).sort_values("article_count", ascending=False) if bad_rows else pd.DataFrame(columns=["token", "article_count"])

    dcnt = daily_counts(df, date_col)
    total = len(df)
    if not dcnt.empty and total:
        peak_count = int(dcnt["article_count"].max())
        peak_ratio = round(peak_count / total * 100, 1)
        peak_date = dcnt.loc[dcnt["article_count"].idxmax(), date_col]
    else:
        peak_count, peak_ratio, peak_date = 0, 0, "-"

    return bad_df, peak_date, peak_count, peak_ratio

def article_table(df, date_col="analysis_date", n=100):
    cols = [c for c in [date_col, "analysis_source", "source_group", "primary_topic", "sentiment_group", "event_detected", "title", "description", "originallink", "link"] if c in df.columns]
    view = df.sort_values(date_col, ascending=False).head(n) if date_col in df.columns else df.head(n)
    st.dataframe(view[cols], use_container_width=True)

# =========================================================
# Load + Compute
# =========================================================

keys = [k for k in list_csv_keys(BUCKET_NAME, S3_PREFIX) if (key_date(k) and key_date(k) >= START_DATE)]
if not keys:
    st.error("S3 processed 경로에서 분석할 CSV 파일을 찾지 못했습니다.")
    st.stop()

df = load_csvs(BUCKET_NAME, sorted(keys))
df, DATE_COL = prepare_df(df)
df = add_topic_columns(df)
df = add_sentiment(df)
df = add_event_tags(df)

latest_date = df[DATE_COL].dropna().max()
latest_df = df[df[DATE_COL] == latest_date].copy()

tfidf_df = tfidf_keyword_scores(df, CORE_KEYWORDS, 25)
analysis_keywords = tfidf_df["keyword"].head(30).tolist() if not tfidf_df.empty else CORE_KEYWORDS
surge_df = surge_sustain_analysis(df, DATE_COL, analysis_keywords)
topic_df = topic_summary(df)
topic_ts_df = topic_timeseries(df, DATE_COL)
net_df = keyword_network(df, analysis_keywords)
daily_df = daily_counts(df, DATE_COL)
bad_token_df, peak_date, peak_count, peak_ratio = data_quality_checks(df, DATE_COL)

# =========================================================
# Tabs
# =========================================================

tabs = st.tabs([
    "1. 분석 개요",
    "2. 데이터 품질 검증",
    "3. TF-IDF 키워드 분석",
    "4. 토픽 분류 분석",
    "5. 기간 내 이슈 변화",
    "6. 이벤트 영향 분석",
    "7. 키워드 관계 분석",
    "8. 리스크 신호 분석",
    "9. 근거 기사"
])

# =========================================================
# 1. Overview
# =========================================================

with tabs[0]:
    st.subheader("분석 개요")
    st.caption("발표에서 설명할 수 있도록 분석 목적, 분석기법, 핵심 결과를 한 화면에 요약합니다.")

    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()
    total_days = df[DATE_COL].nunique()

    top_kw = tfidf_df.iloc[0]["keyword"] if not tfidf_df.empty else "-"
    top_kw_score = tfidf_df.iloc[0]["tfidf_score"] if not tfidf_df.empty else 0
    top_topic = topic_df.iloc[0]["topic"] if not topic_df.empty else "-"
    top_topic_ratio = topic_df.iloc[0]["ratio"] if not topic_df.empty else 0
    top_risk_count = int((df["sentiment_group"] == "부정/리스크").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("분석 기간", f"{min_date} ~ {max_date}", f"총 {total_days:,}일 기준")
    with c2:
        card("분석 기사 수", f"{len(df):,}건", "S3 processed CSV 기준")
    with c3:
        card("핵심 키워드", top_kw, f"TF-IDF 점수 {top_kw_score}")
    with c4:
        card("최대 토픽", top_topic, f"전체 중 {top_topic_ratio}%")

    section("적용한 분석기법")
    m1, m2, m3 = st.columns(3)
    with m1:
        method_box("TF-IDF 키워드 중요도", "단순 빈도가 아니라 분석 기간 내 상대적으로 중요한 IT 키워드 확인", "기사 포함 수 × IDF", "점수가 높을수록 핵심 이슈일 가능성이 높음")
    with m2:
        method_box("시계열/급등률 분석", "키워드가 특정 날짜에 급증했는지 또는 지속적으로 나타났는지 확인", "전일 대비 증가율, 등장 일수", "지속형·급등형·이벤트형으로 구분")
    with m3:
        method_box("동시출현/이벤트 영향 분석", "키워드 간 관계와 행사성 기사 편향 여부 확인", "같은 기사 내 키워드 쌍, 이벤트 키워드 포함률", "트렌드 변화와 단일 이벤트 효과를 구분")

    section("핵심 분석 결과")
    top_sustained = surge_df[surge_df["trend_type"] == "지속형"].head(1)["keyword"].iloc[0] if not surge_df[surge_df["trend_type"] == "지속형"].empty else "-"
    top_surge = surge_df[surge_df["trend_type"] == "급등형"].sort_values("max_growth_rate", ascending=False).head(1)["keyword"].iloc[0] if not surge_df[surge_df["trend_type"] == "급등형"].empty else "-"
    top_pair = f"{net_df.iloc[0]['keyword_a']} ↔ {net_df.iloc[0]['keyword_b']}" if not net_df.empty else "-"
    event_ratio = round(df["is_event_article"].mean() * 100, 1) if len(df) else 0

    insight_box("발표용 요약",
        [
            f"TF-IDF 분석 결과, 전체 기간의 핵심 키워드는 '{top_kw}'로 나타났습니다.",
            f"토픽 분류 결과, 가장 큰 비중을 차지한 주제는 '{top_topic}'이며 전체의 {top_topic_ratio}%입니다.",
            f"급등/지속 분석 결과, 지속형 대표 키워드는 '{top_sustained}', 급등형 대표 키워드는 '{top_surge}'로 확인되었습니다.",
            f"동시출현 분석 결과, 가장 강한 키워드 관계는 '{top_pair}'입니다.",
            f"이벤트 영향 분석 결과, 전체 기사 중 이벤트성 기사 비중은 약 {event_ratio}%로 확인되었습니다."
        ]
    )

    section("연구 질문 재정의")
    st.info("""
    교수님 피드백을 반영하여 '장기 시기별 IT 트렌드 변화'라는 표현은 피하고,
    본 대시보드는 약 3주 내외의 수집 데이터를 기반으로 한 '단기 IT 뉴스 이슈 변화 분석'으로 정의합니다.
    장기 트렌드 분석을 위해서는 수개월~수년 단위의 추가 데이터 수집이 필요합니다.
    """)

# =========================================================
# 2. Data Quality
# =========================================================

with tabs[1]:
    st.subheader("데이터 품질 검증")
    st.caption("분석 결과의 신뢰성을 확인하기 위해 수집 기간, 특정일 쏠림, HTML 잔여 토큰, S3 활용 구조를 점검합니다.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("수집 시작일", df[DATE_COL].min(), "processed CSV 기준")
    with c2:
        card("수집 종료일", df[DATE_COL].max(), "processed CSV 기준")
    with c3:
        card("최대 기사 집중일", peak_date, f"{peak_count:,}건 / 전체 {peak_ratio}%")
    with c4:
        card("S3 파일 수", f"{len(keys):,}개", "Streamlit이 직접 읽은 processed CSV")

    section("S3 기반 분석 파이프라인 검증")
    st.code("""
news-collector-lambda
  → S3 Raw 저장: it_news/IT/raw/YYYY/MM/DD/HH/

news-preprocessor-lambda
  → HTML 정제, Stopword 제거, IT 키워드 필터링, 중복 제거
  → S3 Processed 저장: it_news/IT/processed/news_final_YYYYMMDD.csv

Streamlit Dashboard
  → S3 Processed CSV를 직접 로드하여 분석 수행
""")

    section("일별 데이터 분포")
    st.dataframe(daily_df.sort_values("article_count", ascending=False), use_container_width=True)
    progress_list(daily_df.sort_values("article_count", ascending=False), DATE_COL, "article_count", "기사 수가 많은 날짜 TOP 10", 10)

    section("HTML 엔티티/불용어 잔여 점검")
    if bad_token_df.empty:
        st.success("상위 문제 토큰이 본문/제목에서 탐지되지 않았습니다. 전처리 품질이 개선된 상태로 볼 수 있습니다.")
    else:
        st.warning("아래 토큰이 일부 남아 있습니다. 원본 데이터 또는 전처리 로직을 추가 점검하는 것이 좋습니다.")
        st.dataframe(bad_token_df, use_container_width=True)

    section("분석 한계 명시")
    insight_box("한계점",
        [
            "현재 데이터는 약 3주 내외의 단기 수집 데이터이므로, 수개월~수년 단위의 장기 트렌드 분석으로 일반화하기 어렵습니다.",
            "특정 날짜에 기사가 몰릴 경우 실제 산업 트렌드 변화가 아니라 행사·컨퍼런스·보안 사고 등 이벤트 효과일 수 있습니다.",
            "한국어 형태소 분석기를 완전 적용한 분석은 아니므로, 어절 단위 표현이 일부 남을 수 있습니다.",
            "따라서 본 대시보드의 분석 범위는 '장기 트렌드'가 아니라 '기간 내 뉴스 이슈 변화와 편향 탐지'입니다."
        ]
    )

# =========================================================
# 3. TF-IDF Keyword
# =========================================================

with tabs[2]:
    st.subheader("TF-IDF 키워드 분석")
    method_box("TF-IDF 기반 키워드 중요도", "단순 빈도 상위 키워드가 아니라, 전체 기사 집합 내에서 상대적으로 중요한 키워드를 찾습니다.", "score = article_count × IDF", "빈도는 높지만 모든 기사에 흔한 단어보다, 특정 이슈를 잘 대표하는 키워드가 높게 평가됩니다.")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.dataframe(tfidf_df, use_container_width=True)
    with c2:
        progress_list(tfidf_df, "keyword", "tfidf_score", "TF-IDF 중요도 TOP 10", 10, suffix="점")

    if not tfidf_df.empty:
        selected_kw = st.selectbox("키워드 심층 분석", tfidf_df["keyword"].tolist())
        selected_df = filter_keyword(df, selected_kw)
        selected_daily = selected_df.groupby(DATE_COL).size().reset_index(name="count").sort_values(DATE_COL)

        max_day = selected_daily.loc[selected_daily["count"].idxmax(), DATE_COL] if not selected_daily.empty else "-"
        max_cnt = int(selected_daily["count"].max()) if not selected_daily.empty else 0
        active_days = selected_daily[DATE_COL].nunique()

        section(f"'{selected_kw}' 심층 분석 결과")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            card("전체 언급 기사", f"{len(selected_df):,}건", "전체 기간 기준")
        with k2:
            card("등장 일수", f"{active_days:,}일", "해당 키워드가 등장한 날짜 수")
        with k3:
            card("최고 집중일", max_day, f"{max_cnt:,}건")
        with k4:
            trend_type = surge_df[surge_df["keyword"] == selected_kw]["trend_type"].iloc[0] if not surge_df[surge_df["keyword"] == selected_kw].empty else "미분류"
            card("트렌드 유형", trend_type, "급등/지속 분석 기준")

        insight_box("해석",
            [
                f"'{selected_kw}' 키워드는 전체 기간 동안 {len(selected_df):,}건의 기사에서 등장했습니다.",
                f"가장 많이 등장한 날짜는 {max_day}이며, 해당일 언급량은 {max_cnt:,}건입니다.",
                f"등장 일수와 급등률을 함께 고려하면 이 키워드는 '{trend_type}' 성격으로 해석할 수 있습니다."
            ]
        )
        st.dataframe(selected_daily, use_container_width=True)

# =========================================================
# 4. Topic Analysis
# =========================================================

with tabs[3]:
    st.subheader("토픽 분류 분석")
    method_box("키워드 기반 Topic Classification", "뉴스를 AI, 반도체, 클라우드, 보안 등 주요 IT 주제로 분류하여 관심 분야의 비중을 분석합니다.", "기사 제목+요약문에 포함된 토픽 키워드 매칭", "비중이 높은 토픽은 해당 기간 뉴스 의제에서 더 큰 관심을 받은 분야로 해석합니다.")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(topic_df, use_container_width=True)
    with c2:
        progress_list(topic_df, "topic", "article_count", "토픽별 기사 비중", 10)

    section("토픽별 시계열 변화")
    st.dataframe(topic_ts_df, use_container_width=True)

    if not topic_df.empty:
        top_topic = topic_df.iloc[0]["topic"]
        top_ratio = topic_df.iloc[0]["ratio"]
        insight_box("토픽 분석 결과 해석",
            [
                f"가장 높은 비중의 토픽은 '{top_topic}'이며 전체 기사 중 {top_ratio}%를 차지했습니다.",
                "이는 수집 기간 동안 해당 분야가 IT 뉴스 의제에서 상대적으로 강하게 부각되었음을 의미합니다.",
                "다만 키워드 기반 분류이므로, 향후 KoNLPy 또는 LDA/BERTopic을 적용하면 더 정교한 토픽 모델링이 가능합니다."
            ]
        )

# =========================================================
# 5. Short-term Trend
# =========================================================

with tabs[4]:
    st.subheader("기간 내 이슈 변화 분석")
    method_box("Time-Series + Surge Analysis", "키워드별 일별 언급량을 계산하여 지속적으로 등장한 이슈와 특정 날짜에 급증한 이슈를 구분합니다.", "일별 count, active_days, max_growth_rate", "지속형은 반복 이슈, 급등형은 갑작스러운 관심 증가, 이벤트형은 특정일 집중 이슈로 해석합니다.")

    c1, c2, c3, c4 = st.columns(4)
    sustained_count = len(surge_df[surge_df["trend_type"] == "지속형"]) if not surge_df.empty else 0
    surge_count = len(surge_df[surge_df["trend_type"] == "급등형"]) if not surge_df.empty else 0
    event_count = len(surge_df[surge_df["trend_type"] == "이벤트형"]) if not surge_df.empty else 0
    with c1:
        card("지속형 키워드", f"{sustained_count:,}개", "반복 등장")
    with c2:
        card("급등형 키워드", f"{surge_count:,}개", "전일 대비 증가")
    with c3:
        card("이벤트형 키워드", f"{event_count:,}개", "특정일 집중")
    with c4:
        card("분석 키워드 수", f"{surge_df['keyword'].nunique() if not surge_df.empty else 0:,}개", "TF-IDF 기반 후보")

    selected_type = st.selectbox("트렌드 유형 선택", ["전체", "지속형", "급등형", "이벤트형", "관찰형"])
    view = surge_df if selected_type == "전체" else surge_df[surge_df["trend_type"] == selected_type]
    st.dataframe(view, use_container_width=True)

    if not surge_df.empty:
        growth_top = surge_df.sort_values("max_growth_rate", ascending=False).head(10)
        progress_list(growth_top, "keyword", "max_growth_rate", "급등률 TOP 10", 10, suffix="%")

    insight_box("해석",
        [
            "지속형 키워드는 단기 수집 기간 동안 반복적으로 노출된 핵심 이슈입니다.",
            "급등형 키워드는 특정 날짜에 뉴스 관심이 급격히 증가한 이슈입니다.",
            "이벤트형 키워드는 행사, 발표, 사고 등 단일 이벤트의 영향을 받았을 가능성이 높습니다."
        ]
    )

# =========================================================
# 6. Event Impact
# =========================================================

with tabs[5]:
    st.subheader("이벤트 영향 분석")
    method_box("Event Impact Analysis", "기사량 급증이 일반적인 트렌드 변화인지, 특정 행사나 이벤트 영향인지 구분합니다.", "이벤트 키워드 포함 기사 비율, 급증일 기사량, 이벤트 제외 후 키워드 순위 비교", "이벤트 비중이 높으면 장기 트렌드보다 단일 이벤트 효과로 해석해야 합니다.")

    event_total = int(df["is_event_article"].sum())
    event_ratio = round(event_total / len(df) * 100, 1) if len(df) else 0
    event_daily = df.groupby(DATE_COL)["is_event_article"].agg(["sum", "count"]).reset_index()
    event_daily["event_ratio"] = (event_daily["sum"] / event_daily["count"] * 100).round(1)
    event_daily = event_daily.rename(columns={"sum": "event_article_count", "count": "total_article_count", DATE_COL: "date"})

    c1, c2, c3 = st.columns(3)
    with c1:
        card("이벤트성 기사 수", f"{event_total:,}건", "이벤트 키워드 포함")
    with c2:
        card("이벤트성 기사 비중", f"{event_ratio}%", "전체 기사 대비")
    with c3:
        high_event_date = event_daily.sort_values("event_ratio", ascending=False).iloc[0]["date"] if not event_daily.empty else "-"
        card("이벤트 비중 최고일", high_event_date, "행사성 편향 가능 날짜")

    section("일별 이벤트 영향")
    st.dataframe(event_daily.sort_values("event_ratio", ascending=False), use_container_width=True)

    section("이벤트 기사 제외 전후 핵심 키워드 비교")
    non_event_df = df[~df["is_event_article"]].copy()
    before = tfidf_keyword_scores(df, CORE_KEYWORDS, 15)[["keyword", "article_count", "tfidf_score"]]
    after = tfidf_keyword_scores(non_event_df, CORE_KEYWORDS, 15)[["keyword", "article_count", "tfidf_score"]]

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("### 이벤트 포함")
        st.dataframe(before, use_container_width=True)
    with b2:
        st.markdown("### 이벤트 제외")
        st.dataframe(after, use_container_width=True)

    insight_box("해석",
        [
            f"전체 기사 중 이벤트성 기사는 {event_total:,}건으로 약 {event_ratio}%입니다.",
            "이벤트 제외 전후 키워드 순위가 크게 달라진다면, 해당 기간의 뉴스 흐름은 일반 트렌드보다 행사성 보도에 영향을 받은 것으로 볼 수 있습니다.",
            "이 분석은 교수님 피드백의 '단일 이벤트 효과와 트렌드 변화를 구분해야 한다'는 지점을 보완합니다."
        ]
    )

# =========================================================
# 7. Relationship
# =========================================================

with tabs[6]:
    st.subheader("키워드 관계 분석")
    method_box("Co-occurrence Network Analysis", "같은 기사 안에서 함께 등장한 키워드 쌍을 분석하여 IT 이슈 간 연결 구조를 파악합니다.", "기사별 등장 키워드 조합 → 동시출현 빈도 계산", "동시출현 빈도가 높을수록 두 이슈가 같은 맥락에서 보도된 것으로 해석합니다.")

    if net_df.empty:
        st.warning("키워드 관계를 계산할 데이터가 부족합니다.")
    else:
        top_pair = net_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            card("최강 연결", f"{top_pair['keyword_a']} ↔ {top_pair['keyword_b']}", f"{int(top_pair['co_count']):,}건 동시출현")
        with c2:
            card("관계 수", f"{len(net_df):,}개", "키워드 페어 기준")
        with c3:
            nodes = len(set(net_df["keyword_a"]).union(set(net_df["keyword_b"])))
            card("연결 키워드 수", f"{nodes:,}개", "네트워크 노드 기준")

        components.html(network_html(net_df), height=720, scrolling=True)
        st.dataframe(net_df.head(30), use_container_width=True)

        insight_box("해석",
            [
                f"가장 강한 연결은 '{top_pair['keyword_a']} ↔ {top_pair['keyword_b']}'입니다.",
                "이는 두 키워드가 같은 기사 안에서 자주 함께 등장했다는 의미입니다.",
                "따라서 단순히 개별 키워드가 많이 나온 것이 아니라, 이슈 간 관계 구조를 확인할 수 있습니다."
            ]
        )

# =========================================================
# 8. Risk
# =========================================================

with tabs[7]:
    st.subheader("리스크 신호 분석")
    method_box("Rule-based Sentiment/Risk Analysis", "성장·투자·출시 등 긍정 키워드와 해킹·유출·취약점 등 리스크 키워드를 비교해 보도 성향을 분류합니다.", "긍정 키워드 수 vs 리스크 키워드 수", "리스크 비중이 높은 토픽은 보안 사고나 규제 이슈 가능성이 높습니다.")

    sentiment_count = df["sentiment_group"].value_counts().reset_index()
    sentiment_count.columns = ["sentiment", "count"]
    sentiment_count["ratio"] = (sentiment_count["count"] / len(df) * 100).round(1)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(sentiment_count, use_container_width=True)
    with c2:
        progress_list(sentiment_count, "sentiment", "count", "감성/리스크 분포", 5)

    topic_risk = df.pivot_table(index="primary_topic", columns="sentiment_group", values="title", aggfunc="count", fill_value=0).reset_index()
    total_cols = [c for c in ["긍정/성장", "중립", "부정/리스크"] if c in topic_risk.columns]
    topic_risk["total"] = topic_risk[total_cols].sum(axis=1)
    if "부정/리스크" not in topic_risk.columns:
        topic_risk["부정/리스크"] = 0
    topic_risk["risk_ratio"] = (topic_risk["부정/리스크"] / topic_risk["total"] * 100).round(1)
    topic_risk = topic_risk.sort_values("risk_ratio", ascending=False)

    section("토픽별 리스크 비율")
    st.dataframe(topic_risk, use_container_width=True)

    risk_df = df[df["sentiment_group"] == "부정/리스크"]
    insight_box("해석",
        [
            f"전체 기사 중 리스크 성향 기사는 {len(risk_df):,}건입니다.",
            "리스크 비율이 높은 토픽은 해킹, 개인정보, 취약점, 장애 등 부정적 이슈와 더 밀접하게 연결되어 있습니다.",
            "이를 통해 IT 뉴스 안에서 성장 이슈와 위험 신호를 분리해 확인할 수 있습니다."
        ]
    )

    with st.expander("리스크 기사 근거 보기"):
        article_table(risk_df, DATE_COL, n=100)

# =========================================================
# 9. Evidence
# =========================================================

with tabs[8]:
    st.subheader("근거 기사")
    st.caption("분석 결과의 근거가 되는 실제 기사 데이터를 확인하는 탭입니다. 앞쪽 탭은 분석 결과, 이 탭은 검증용 근거 데이터입니다.")

    col1, col2, col3 = st.columns(3)
    with col1:
        topic_filter = st.selectbox("토픽 필터", ["전체"] + sorted(df["primary_topic"].dropna().unique().tolist()))
    with col2:
        sentiment_filter = st.selectbox("성향 필터", ["전체"] + sorted(df["sentiment_group"].dropna().unique().tolist()))
    with col3:
        keyword_filter = st.text_input("키워드 검색", "")

    view = df.copy()
    if topic_filter != "전체":
        view = view[view["primary_topic"] == topic_filter]
    if sentiment_filter != "전체":
        view = view[view["sentiment_group"] == sentiment_filter]
    if keyword_filter.strip():
        view = filter_keyword(view, keyword_filter.strip())

    st.info(f"필터 적용 결과: {len(view):,}건")
    article_table(view, DATE_COL, n=300)
