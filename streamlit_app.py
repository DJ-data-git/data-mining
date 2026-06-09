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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# Page / Style
# =========================================================

st.set_page_config(page_title="IT 뉴스 데이터마이닝 분석 대시보드", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a 0%,#111827 45%,#020617 100%); color:#e5e7eb;}
.block-container {padding-top:1.8rem; padding-bottom:3rem;}
h1 {color:#f8fafc; font-weight:900; letter-spacing:-0.05em;}
h2,h3 {color:#e5e7eb; font-weight:850;}
[data-testid="stCaptionContainer"] {color:#94a3b8;}
[data-testid="stDataFrame"] {border-radius:16px; overflow:hidden; border:1px solid rgba(148,163,184,.18);}
.stAlert {background:rgba(14,165,233,.12); border:1px solid rgba(56,189,248,.35); border-radius:16px; color:#e0f2fe;}
.stButton > button {background:linear-gradient(135deg,#0284c7,#2563eb); color:white; border:0; border-radius:999px; padding:.5rem 1rem; font-weight:700;}
.stTextInput input, .stSelectbox div[data-baseweb="select"] {background:#020617; color:#e5e7eb; border-radius:12px;}
hr {border:none; height:1px; background:linear-gradient(90deg,transparent,#38bdf8,transparent); margin:2rem 0;}
section[data-testid="stSidebar"] {background:#020617;}
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

st.title("IT 뉴스 데이터마이닝 분석 대시보드")
st.caption("AWS Lambda로 수집·전처리한 IT 뉴스 데이터를 S3에서 불러와 TF-IDF, 시계열, 감성, 동시출현, LDA 토픽 모델링으로 분석합니다.")

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

# =========================================================
# Dictionaries
# =========================================================

CORE_KEYWORDS = [
    "AI", "인공지능", "생성형AI", "생성형 AI", "LLM", "챗GPT", "ChatGPT", "GPT", "OpenAI",
    "반도체", "AI반도체", "AI 반도체", "HBM", "GPU", "엔비디아", "NVIDIA",
    "삼성전자", "SK하이닉스", "파운드리", "메모리", "칩",
    "클라우드", "AWS", "Azure", "데이터센터", "데이터 센터", "서버", "인프라", "SaaS",
    "쿠버네티스", "Kubernetes",
    "보안", "해킹", "개인정보", "랜섬웨어", "침해", "취약점", "사이버", "정보보호",
    "로봇", "전기차", "배터리", "자율주행", "모빌리티", "이차전지",
    "네이버", "카카오", "구글", "애플", "메타", "마이크로소프트", "MS",
    "플랫폼", "빅테크", "데이터", "빅데이터", "소프트웨어", "디지털", "DX",
    "핀테크", "블록체인", "가상자산", "메타버스"
]

TOPIC_MAP = {
    "AI/인공지능": ["AI", "인공지능", "생성형AI", "챗GPT", "ChatGPT", "GPT", "OpenAI", "LLM", "GPU", "엔비디아"],
    "반도체": ["반도체", "AI반도체", "AI 반도체", "HBM", "삼성전자", "SK하이닉스", "파운드리", "메모리", "칩"],
    "클라우드/인프라": ["클라우드", "AWS", "Azure", "데이터센터", "데이터 센터", "서버", "인프라", "SaaS", "쿠버네티스"],
    "보안/리스크": ["보안", "해킹", "개인정보", "랜섬웨어", "침해", "취약점", "사이버", "정보보호"],
    "모빌리티/로봇": ["전기차", "자율주행", "배터리", "로봇", "모빌리티", "이차전지"],
    "플랫폼/빅테크": ["네이버", "카카오", "구글", "애플", "메타", "마이크로소프트", "MS", "플랫폼", "빅테크"],
}

POSITIVE_WORDS = [
    "성장", "확대", "출시", "투자", "협력", "개선", "강화", "수주", "증가", "성공",
    "최초", "고도화", "혁신", "개발", "도입", "선정", "공급", "상승", "기대"
]
NEGATIVE_WORDS = [
    "해킹", "침해", "유출", "장애", "중단", "규제", "감소", "적자", "위험", "논란",
    "피해", "취약점", "공격", "랜섬웨어", "오류", "실패", "제재", "우려", "하락"
]
RISK_KEYWORDS = ["보안", "해킹", "개인정보", "랜섬웨어", "침해", "취약점", "사이버", "유출", "공격"]

STOPWORDS = [
    "the", "and", "for", "that", "with", "this", "from", "have", "will", "into", "about",
    "their", "they", "them", "were", "been", "being", "said", "more", "than", "over",
    "after", "before", "while", "where", "when", "what", "which", "would", "could", "should",
    "there", "these", "those", "because", "through", "during", "under", "between", "among",
    "it", "to", "of", "in", "is", "on", "at", "by", "be", "as", "an", "or", "if", "we",
    "he", "she", "you", "are", "was", "has", "had", "can", "not", "also", "but", "how",
    "may", "assumed", "errors", "translation", "translated",
    "뉴스", "기사", "기자", "보도", "관련", "대한", "위해", "통해", "이번", "최근", "오늘",
    "있다", "했다", "한다", "밝혔다", "설명했다", "말했다", "지난", "올해", "현재", "가운데",
    "것", "수", "등", "및", "일", "월", "년", "전", "후", "중", "개", "명"
]

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
    border-radius:20px;padding:20px;min-height:136px;box-shadow:0 0 20px rgba(56,189,248,.06);">
      <div style="color:#94a3b8;font-size:13px;font-weight:750;margin-bottom:8px;">{title}</div>
      <div style="color:{color};font-size:27px;font-weight:950;margin-bottom:8px;line-height:1.25;">{value}</div>
      <div style="color:#cbd5e1;font-size:13px;line-height:1.55;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def method_box(method, purpose, calculation, output):
    st.markdown(f"""
    <div style="background:rgba(2,6,23,.74);border:1px solid rgba(148,163,184,.18);
    border-radius:18px;padding:17px;margin-bottom:12px;">
      <div style="font-size:18px;font-weight:950;color:#f8fafc;margin-bottom:10px;">분석기법: {method}</div>
      <div style="color:#cbd5e1;line-height:1.7;"><b>분석 목적</b>: {purpose}</div>
      <div style="color:#cbd5e1;line-height:1.7;"><b>계산 방식</b>: {calculation}</div>
      <div style="color:#e0f2fe;line-height:1.7;"><b>도출 결과</b>: {output}</div>
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
        label = row[label_col]
        value = float(row[value_col])
        ratio = value / max_v * 100 if max_v else 0
        value_text = f"{int(value):,}{suffix}" if value == int(value) else f"{value:.2f}{suffix}"
        st.markdown(f"""
        <div style="background:rgba(15,23,42,.75);border:1px solid rgba(148,163,184,.12);border-radius:16px;
        padding:13px 16px;margin-bottom:9px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;">
            <div style="font-size:15px;font-weight:850;color:white;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{label}</div>
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

# =========================================================
# Data Utilities
# =========================================================

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
        df[col] = df[col].fillna("").astype(str)

    df.loc[df["source"].str.strip() == "", "source"] = "unknown"
    df.loc[df["source_group"].str.strip() == "", "source_group"] = df["source"]
    df["analysis_source"] = df["source"]

    dedup_cols = [c for c in ["originallink", "link", "title"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last")

    return df, date_col


def text_series(df):
    return df["title"].fillna("").astype(str) + " " + df["description"].fillna("").astype(str)


def clean_for_vectorize(text):
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^가-힣A-Za-z0-9\s+#.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_any(text, keywords):
    text = str(text).lower()
    return any(str(kw).lower() in text for kw in keywords)


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
    if not rows:
        return pd.DataFrame(columns=["keyword", "count"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def add_period_column(df, date_col, period_option):
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce")

    if period_option == "일별":
        out["analysis_period"] = dt.dt.strftime("%Y-%m-%d")
        out["analysis_period_label"] = out["analysis_period"]

    elif period_option == "주별":
        week_start = dt - pd.to_timedelta(dt.dt.weekday, unit="D")
        week_end = week_start + pd.Timedelta(days=6)
        out["analysis_period"] = week_start.dt.strftime("%Y-%m-%d")
        out["analysis_period_label"] = week_start.dt.strftime("%Y-%m-%d") + " ~ " + week_end.dt.strftime("%Y-%m-%d")

    elif period_option == "월별":
        out["analysis_period"] = dt.dt.strftime("%Y-%m")
        out["analysis_period_label"] = dt.dt.year.astype("Int64").astype(str) + "년 " + dt.dt.month.astype("Int64").astype(str) + "월"

    elif period_option == "연별":
        out["analysis_period"] = dt.dt.strftime("%Y")
        out["analysis_period_label"] = dt.dt.year.astype("Int64").astype(str) + "년"

    else:
        out["analysis_period"] = out[date_col].astype(str)
        out["analysis_period_label"] = out["analysis_period"]

    out["analysis_period"] = out["analysis_period"].fillna("unknown").astype(str)
    out["analysis_period_label"] = out["analysis_period_label"].fillna(out["analysis_period"]).astype(str)
    out.loc[out["analysis_period_label"].str.contains("<NA>", na=False), "analysis_period_label"] = "unknown"

    return out


def get_current_period_info(df, date_col, period_option):
    valid_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if valid_dates.empty:
        return "-", "-"
    latest_dt = valid_dates.max()
    temp = pd.DataFrame({date_col: [latest_dt.strftime("%Y-%m-%d")]})
    temp = add_period_column(temp, date_col, period_option)
    return temp["analysis_period"].iloc[0], temp["analysis_period_label"].iloc[0]


def classify_topic(text):
    for topic, kws in TOPIC_MAP.items():
        if contains_any(text, kws):
            return topic
    return "기타"


def add_topic(df):
    out = df.copy()
    out["primary_topic"] = text_series(out).apply(classify_topic)
    return out


def classify_sentiment(text):
    pos = sum(w in str(text) for w in POSITIVE_WORDS)
    neg = sum(w in str(text) for w in NEGATIVE_WORDS)

    if neg > pos:
        return "부정/리스크"
    if pos > neg:
        return "긍정/성장"
    return "중립"


def add_sentiment(df):
    out = df.copy()
    out["sentiment_group"] = text_series(out).apply(classify_sentiment)
    return out


def topic_counts(df):
    if df.empty or "primary_topic" not in df.columns:
        return pd.DataFrame(columns=["topic", "count", "ratio"])
    res = df["primary_topic"].value_counts().reset_index()
    res.columns = ["topic", "count"]
    res["ratio"] = (res["count"] / len(df) * 100).round(1) if len(df) else 0
    return res


def daily_counts(df, date_col):
    return df.groupby(date_col).size().reset_index(name="article_count").sort_values(date_col)


def period_counts(df, period_col, period_label_map):
    res = df.groupby(period_col).size().reset_index(name="article_count").sort_values(period_col)
    res["period_label"] = res[period_col].map(period_label_map).fillna(res[period_col])
    return res


# =========================================================
# Analysis Functions
# =========================================================

def tfidf_keywords(df, top_n=25):
    docs = text_series(df).map(clean_for_vectorize).tolist()
    docs = [d for d in docs if d.strip()]

    if len(docs) < 2:
        return pd.DataFrame(columns=["keyword", "tfidf_score", "doc_count"])

    vectorizer = TfidfVectorizer(
        max_features=1500,
        stop_words=STOPWORDS,
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9+#.-]{2,}\b"
    )

    try:
        mat = vectorizer.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(columns=["keyword", "tfidf_score", "doc_count"])

    scores = mat.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    doc_counts = (mat > 0).sum(axis=0).A1

    result = pd.DataFrame({
        "keyword": terms,
        "tfidf_score": scores,
        "doc_count": doc_counts
    })
    result = result[~result["keyword"].str.lower().isin([s.lower() for s in STOPWORDS])]
    return result.sort_values("tfidf_score", ascending=False).head(top_n).reset_index(drop=True)


def keyword_candidate_tfidf(df, keywords=CORE_KEYWORDS, top_n=25):
    txt = text_series(df)
    total_docs = len(txt)
    rows = []

    for kw in keywords:
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
            "score": round(score, 4),
            "coverage_ratio": round(article_count / total_docs * 100, 2) if total_docs else 0
        })

    if not rows:
        return pd.DataFrame(columns=["keyword", "article_count", "idf", "score", "coverage_ratio"])

    return pd.DataFrame(rows).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def similar_articles(df, keyword, top_n=10):
    txt = text_series(df).fillna("").astype(str)
    if len(txt) < 2:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(
        max_features=1200,
        stop_words=STOPWORDS,
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9+#.-]{2,}\b"
    )

    try:
        mat = vectorizer.fit_transform(txt.tolist() + [keyword])
        scores = cosine_similarity(mat[:-1], mat[-1]).flatten()
    except ValueError:
        return pd.DataFrame()

    out = df.copy()
    out["similarity_score"] = scores
    cols = [c for c in ["analysis_date", "analysis_source", "primary_topic", "sentiment_group", "title", "description", "originallink", "link", "similarity_score"] if c in out.columns]
    return out.sort_values("similarity_score", ascending=False)[cols].head(top_n)


def period_keyword_change(current_df, previous_df, keywords):
    cur = keyword_counts(current_df, keywords).rename(columns={"count": "current_count"})
    prev = keyword_counts(previous_df, keywords).rename(columns={"count": "previous_count"})

    merged = pd.merge(cur, prev, on="keyword", how="outer").fillna(0)
    merged["current_count"] = merged["current_count"].astype(int)
    merged["previous_count"] = merged["previous_count"].astype(int)
    merged["change"] = merged["current_count"] - merged["previous_count"]
    merged["growth_rate"] = merged.apply(
        lambda r: 100.0 if r["previous_count"] == 0 and r["current_count"] > 0
        else round(r["change"] / r["previous_count"] * 100, 1) if r["previous_count"] > 0
        else 0.0,
        axis=1
    )
    return merged.sort_values(["change", "current_count"], ascending=False).reset_index(drop=True)


def keyword_timeseries(df, date_col, keywords):
    rows = []
    for date in sorted(df[date_col].dropna().unique()):
        sub = df[df[date_col] == date]
        counts = keyword_counts(sub, keywords)
        for _, row in counts.iterrows():
            rows.append({"date": date, "keyword": row["keyword"], "count": int(row["count"])})
    return pd.DataFrame(rows)


def keyword_network(df, keywords):
    rows = []
    for text in text_series(df):
        lower = str(text).lower()
        appeared = sorted({kw for kw in keywords if str(kw).lower() in lower})
        rows.extend(combinations(appeared, 2))
    if not rows:
        return pd.DataFrame(columns=["keyword_a", "keyword_b", "co_count"])
    return pd.DataFrame(rows, columns=["keyword_a", "keyword_b"]).value_counts().reset_index(name="co_count").sort_values("co_count", ascending=False)


def network_html(net_df, top_n=30):
    G = nx.Graph()
    for _, row in net_df.head(top_n).iterrows():
        weight = int(row["co_count"])
        G.add_edge(
            row["keyword_a"], row["keyword_b"],
            value=max(1, weight),
            title=f"{row['keyword_a']} ↔ {row['keyword_b']} | 동시출현 {weight:,}건"
        )

    net = Network(height="680px", width="100%", bgcolor="#0f172a", font_color="white", notebook=False)
    net.from_nx(G)

    weighted_degree = dict(G.degree(weight="value"))
    degrees = dict(G.degree())

    for node in net.nodes:
        node_id = node["id"]
        node["size"] = min(76, max(24, 20 + weighted_degree.get(node_id, 1) * 0.6))
        node["color"] = {"background": "#38bdf8", "border": "#7dd3fc"}
        node["font"] = {"size": 18, "face": "Arial", "color": "#f8fafc", "strokeWidth": 3, "strokeColor": "#0f172a"}
        node["title"] = f"{node_id}<br>연결 키워드 수: {degrees.get(node_id, 0)}<br>연결 강도: {weighted_degree.get(node_id, 0):.1f}"

    for edge in net.edges:
        edge["width"] = min(10, max(1, edge.get("value", 1) / 5))
        edge["color"] = {"color": "rgba(148,163,184,.72)", "highlight": "#38bdf8"}

    net.set_options("""
    var options = {
      "interaction": {"hover": true, "tooltipDelay": 120, "dragNodes": true, "dragView": true, "zoomView": true},
      "nodes": {"shape": "dot"},
      "edges": {"smooth": {"enabled": true, "type": "continuous"}},
      "physics": {"enabled": false}
    }
    """)

    path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    net.save_graph(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@st.cache_data(ttl=600)
def lda_topic_modeling(docs, n_topics=5, n_words=8, max_features=1200):
    cleaned_docs = [clean_for_vectorize(d) for d in docs if str(d).strip()]

    if len(cleaned_docs) < n_topics:
        return pd.DataFrame(), pd.DataFrame()

    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words=STOPWORDS,
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9+#.-]{2,}\b",
        min_df=2
    )

    try:
        doc_term = vectorizer.fit_transform(cleaned_docs)
    except ValueError:
        return pd.DataFrame(), pd.DataFrame()

    if doc_term.shape[0] < n_topics or doc_term.shape[1] < n_topics:
        return pd.DataFrame(), pd.DataFrame()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
        max_iter=20
    )

    topic_matrix = lda.fit_transform(doc_term)
    terms = vectorizer.get_feature_names_out()

    topic_rows = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        top_indices = topic_weights.argsort()[::-1][:n_words]
        top_terms = [terms[i] for i in top_indices]
        topic_rows.append({
            "topic_id": f"Topic {topic_idx + 1}",
            "top_words": ", ".join(top_terms),
            "weight_sum": round(float(topic_weights[top_indices].sum()), 2)
        })

    doc_topics = pd.DataFrame(topic_matrix, columns=[f"Topic {i+1}" for i in range(n_topics)])
    doc_topics["dominant_topic"] = doc_topics.idxmax(axis=1)
    doc_topics["dominant_score"] = doc_topics[[f"Topic {i+1}" for i in range(n_topics)]].max(axis=1).round(4)

    topic_summary = doc_topics["dominant_topic"].value_counts().reset_index()
    topic_summary.columns = ["topic_id", "article_count"]

    topic_df = pd.DataFrame(topic_rows)
    topic_df = pd.merge(topic_df, topic_summary, on="topic_id", how="left").fillna({"article_count": 0})
    topic_df["article_count"] = topic_df["article_count"].astype(int)
    topic_df["ratio"] = (topic_df["article_count"] / len(doc_topics) * 100).round(1)

    return topic_df, doc_topics


def article_table(df, date_col="analysis_date", n=200):
    cols = [c for c in [date_col, "analysis_source", "source_group", "primary_topic", "sentiment_group", "title", "description", "originallink", "link"] if c in df.columns]
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
df = add_topic(df)
df = add_sentiment(df)

latest_date = df[DATE_COL].dropna().max()
latest_df = df[df[DATE_COL] == latest_date].copy()

# =========================================================
# Sidebar Global Filter
# =========================================================

st.sidebar.header("분석 기준 설정")

period_option = st.sidebar.selectbox(
    "분석 단위 선택",
    ["일별", "주별", "월별", "연별"],
    index=0,
    help="선택한 단위에 따라 Summary와 시계열 분석의 기준 구간이 바뀝니다."
)

summary_df = add_period_column(df, DATE_COL, period_option)
PERIOD_COL = "analysis_period"
PERIOD_LABEL_COL = "analysis_period_label"

period_label_map = (
    summary_df[[PERIOD_COL, PERIOD_LABEL_COL]]
    .drop_duplicates()
    .set_index(PERIOD_COL)[PERIOD_LABEL_COL]
    .to_dict()
)

periods = sorted(summary_df[PERIOD_COL].dropna().unique().tolist())
current_period_value, current_period_label = get_current_period_info(df, DATE_COL, period_option)
current_period = current_period_value if current_period_value in periods else (periods[-1] if periods else None)
previous_period = periods[-2] if len(periods) >= 2 else None

active_df = summary_df[summary_df[PERIOD_COL] == current_period].copy() if current_period else summary_df.head(0).copy()
previous_df = summary_df[summary_df[PERIOD_COL] == previous_period].copy() if previous_period else summary_df.head(0).copy()

st.sidebar.info(
    f"현재 분석 구간: {current_period_label}\n\n"
    f"현재 구간 기사 수: {len(active_df):,}건\n\n"
    f"전체 기사 수: {len(df):,}건"
)

# Current/overall metrics
active_keyword_count = keyword_counts(active_df, CORE_KEYWORDS)
overall_keyword_count = keyword_counts(df, CORE_KEYWORDS)
active_tfidf = tfidf_keywords(active_df, top_n=20)
overall_tfidf = tfidf_keywords(df, top_n=25)
candidate_tfidf = keyword_candidate_tfidf(active_df, CORE_KEYWORDS, top_n=20)
active_topics = topic_counts(active_df)
active_sentiment = active_df["sentiment_group"].value_counts().reset_index()
active_sentiment.columns = ["sentiment", "count"] if not active_sentiment.empty else ["sentiment", "count"]
keyword_change_df = period_keyword_change(active_df, previous_df, CORE_KEYWORDS)

network_keywords = candidate_tfidf["keyword"].head(25).tolist() if not candidate_tfidf.empty else overall_keyword_count["keyword"].head(25).tolist()
net_df = keyword_network(active_df if len(active_df) else df, network_keywords)
lda_docs = text_series(active_df if len(active_df) >= 20 else df).tolist()

# =========================================================
# Tabs
# =========================================================

tab_summary, tab_tfidf, tab_time, tab_sentiment, tab_network, tab_lda, tab_evidence = st.tabs([
    "1. Executive Summary",
    "2. TF-IDF Analysis",
    "3. Time Series Analysis",
    "4. Sentiment Analysis",
    "5. Co-occurrence Analysis",
    "6. LDA Topic Modeling",
    "7. Evidence Center"
])

# =========================================================
# 1. Executive Summary
# =========================================================

with tab_summary:
    st.subheader("Executive Summary")
    st.caption("선택한 분석 단위의 현재 구간을 기준으로 기사량, 핵심 키워드, 주제, 감성, 관계 분석 결과를 요약합니다.")

    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()
    total_periods = summary_df[PERIOD_COL].nunique()
    active_total = len(active_df)

    top_keyword = active_keyword_count.iloc[0]["keyword"] if not active_keyword_count.empty else "-"
    top_keyword_count = int(active_keyword_count.iloc[0]["count"]) if not active_keyword_count.empty else 0

    top_topic = active_topics.iloc[0]["topic"] if not active_topics.empty else "-"
    top_topic_count = int(active_topics.iloc[0]["count"]) if not active_topics.empty else 0

    risk_df = filter_keywords(active_df, RISK_KEYWORDS)
    risk_ratio = round(len(risk_df) / active_total * 100, 1) if active_total else 0

    top_pair = f"{net_df.iloc[0]['keyword_a']} ↔ {net_df.iloc[0]['keyword_b']}" if not net_df.empty else "-"
    top_pair_count = int(net_df.iloc[0]["co_count"]) if not net_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("현재 분석 구간", current_period_label, f"원본 범위 {min_date} ~ {max_date} / {period_option} 기준 {total_periods:,}개 구간")
    with c2:
        card("현재 구간 기사 수", f"{active_total:,}건", f"전체 수집 기사 {len(df):,}건 중 현재 구간")
    with c3:
        card("현재 핵심 키워드", top_keyword, f"{top_keyword_count:,}건 언급")
    with c4:
        card("리스크 기사 비중", f"{risk_ratio}%", f"리스크 키워드 기사 {len(risk_df):,}건", "#ef4444")

    section("적용 분석기법")
    m1, m2, m3 = st.columns(3)
    with m1:
        method_box("TF-IDF Analysis", "단순 빈도보다 해당 기간을 대표하는 중요 키워드 추출", "문서별 TF-IDF 점수 합산", "핵심 키워드와 관련 기사 도출")
    with m2:
        method_box("Time Series Analysis", "일/주/월/연 기준 뉴스량과 키워드 변화 확인", "기간별 기사 수 및 직전 구간 대비 변화량 계산", "증가 이슈와 감소 이슈 파악")
    with m3:
        method_box("LDA Topic Modeling", "뉴스 집합 내부의 잠재 토픽 탐색", "CountVectorizer + Latent Dirichlet Allocation", "토픽별 대표 단어와 기사 비중 도출")

    section(f"현재 {period_option} 구간 분석 결과")
    col_a, col_b = st.columns(2)
    with col_a:
        progress_list(active_keyword_count.head(10), "keyword", "count", f"{current_period_label} 핵심 키워드 TOP 10")
    with col_b:
        progress_list(active_topics.head(10), "topic", "count", f"{current_period_label} 주제 분포")

    section("직전 구간 대비 증가 키워드")
    progress_list(keyword_change_df.head(10), "keyword", "change", "증가량 TOP 10", suffix="건")

    insight_box("요약 해석", [
        f"현재 분석 구간은 {current_period_label}이며, 해당 구간의 기사 수는 {active_total:,}건입니다.",
        f"현재 구간의 최다 키워드는 '{top_keyword}'이며 {top_keyword_count:,}건 등장했습니다.",
        f"현재 구간의 최다 주제는 '{top_topic}'이며 {top_topic_count:,}건 관련 기사가 확인되었습니다.",
        f"리스크 키워드가 포함된 기사는 {len(risk_df):,}건이며, 전체 대비 {risk_ratio}%입니다.",
        f"동시출현 분석 기준 가장 강한 관계는 '{top_pair}'이며 {top_pair_count:,}건 함께 등장했습니다."
    ])

    with st.expander("현재 구간 필터 검증"):
        st.write("현재 분석 단위:", period_option)
        st.write("현재 분석 구간:", current_period_label)
        st.write("active_df 기사 수:", len(active_df))
        st.dataframe(active_df[[DATE_COL, PERIOD_COL, PERIOD_LABEL_COL, "title"]].head(20), use_container_width=True)

# =========================================================
# 2. TF-IDF
# =========================================================

with tab_tfidf:
    st.subheader("TF-IDF Analysis")
    method_box(
        "TF-IDF Analysis",
        "현재 분석 구간에서 단순히 자주 등장한 단어가 아니라, 해당 뉴스 집합을 상대적으로 잘 대표하는 키워드를 추출합니다.",
        "TfidfVectorizer를 적용하여 각 단어의 중요도를 계산한 뒤 전체 문서 기준으로 점수를 합산합니다.",
        "TF-IDF 점수가 높은 키워드는 현재 구간의 특징적인 이슈로 해석합니다."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 전체 단어 기반 TF-IDF")
        st.dataframe(active_tfidf, use_container_width=True)
    with col2:
        st.markdown("### IT 후보 키워드 기반 TF-IDF")
        st.dataframe(candidate_tfidf, use_container_width=True)

    if not candidate_tfidf.empty:
        selected_kw = st.selectbox("키워드 심층 분석", candidate_tfidf["keyword"].tolist())
        selected_df = filter_keyword(active_df, selected_kw)
        selected_all_df = filter_keyword(df, selected_kw)

        k1, k2, k3 = st.columns(3)
        with k1:
            card("현재 구간 관련 기사", f"{len(selected_df):,}건", current_period_label)
        with k2:
            card("전체 기간 관련 기사", f"{len(selected_all_df):,}건", "전체 수집 범위")
        with k3:
            score = candidate_tfidf[candidate_tfidf["keyword"] == selected_kw]["score"].iloc[0]
            card("TF-IDF 점수", f"{score}", "후보 키워드 기준")

        section(f"'{selected_kw}' 관련 기사")
        article_table(selected_df, DATE_COL, n=100)

        section(f"'{selected_kw}' 유사 기사")
        st.dataframe(similar_articles(active_df, selected_kw, top_n=10), use_container_width=True)

# =========================================================
# 3. Time Series
# =========================================================

with tab_time:
    st.subheader("Time Series Analysis")
    method_box(
        "Time Series Analysis",
        "뉴스 기사량과 키워드가 시간에 따라 어떻게 변화하는지 확인합니다.",
        "일별/주별/월별/연별로 데이터를 재집계하고, 현재 구간과 직전 구간의 키워드 빈도를 비교합니다.",
        "기간별 기사량 변화와 급상승 키워드를 통해 단기 이슈 변화를 파악합니다."
    )

    period_count_df = period_counts(summary_df, PERIOD_COL, period_label_map)

    c1, c2 = st.columns([1, 1])
    with c1:
        progress_list(period_count_df.sort_values("article_count", ascending=False), "period_label", "article_count", f"{period_option} 기사량 TOP 10")
    with c2:
        st.dataframe(period_count_df, use_container_width=True)

    section("직전 구간 대비 키워드 변화")
    st.dataframe(keyword_change_df, use_container_width=True)
    progress_list(keyword_change_df.head(10), "keyword", "change", "증가 키워드 TOP 10", suffix="건")

    section("일별 키워드 변화")
    kw_for_ts = st.selectbox("시계열 확인 키워드", overall_keyword_count["keyword"].head(30).tolist() if not overall_keyword_count.empty else CORE_KEYWORDS)
    ts_df = keyword_timeseries(df, DATE_COL, [kw_for_ts])
    if ts_df.empty:
        st.warning("선택 키워드의 시계열 데이터가 없습니다.")
    else:
        st.dataframe(ts_df, use_container_width=True)

    insight_box("해석", [
        f"현재 대시보드는 '{period_option}' 기준으로 데이터를 재집계합니다.",
        "기간별 기사량은 특정 시점에 뉴스가 집중되는 구간을 확인하는 데 사용합니다.",
        "직전 구간 대비 증가량이 큰 키워드는 단기적으로 관심이 증가한 이슈로 해석할 수 있습니다."
    ])

# =========================================================
# 4. Sentiment
# =========================================================

with tab_sentiment:
    st.subheader("Sentiment Analysis")
    method_box(
        "Rule-based Sentiment Analysis",
        "뉴스 제목과 요약문에 포함된 긍정/부정 키워드를 기준으로 기사 성향을 분류합니다.",
        "긍정 키워드 수와 부정/리스크 키워드 수를 비교하여 긍정/성장, 중립, 부정/리스크로 분류합니다.",
        "IT 뉴스 내 성장 이슈와 위험 신호의 비중을 비교합니다."
    )

    sentiment_df = active_df["sentiment_group"].value_counts().reset_index()
    sentiment_df.columns = ["sentiment", "count"]
    sentiment_df["ratio"] = (sentiment_df["count"] / len(active_df) * 100).round(1) if len(active_df) else 0

    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(sentiment_df, use_container_width=True)
    with c2:
        progress_list(sentiment_df, "sentiment", "count", "현재 구간 감성 분포")

    section("주제별 감성 분포")
    if len(active_df):
        topic_sentiment = active_df.pivot_table(
            index="primary_topic",
            columns="sentiment_group",
            values="title",
            aggfunc="count",
            fill_value=0
        ).reset_index()
        sentiment_cols = [c for c in ["긍정/성장", "중립", "부정/리스크"] if c in topic_sentiment.columns]
        topic_sentiment["total"] = topic_sentiment[sentiment_cols].sum(axis=1) if sentiment_cols else 0
        if "부정/리스크" not in topic_sentiment.columns:
            topic_sentiment["부정/리스크"] = 0
        topic_sentiment["risk_ratio"] = (topic_sentiment["부정/리스크"] / topic_sentiment["total"] * 100).round(1).fillna(0)
        st.dataframe(topic_sentiment.sort_values("risk_ratio", ascending=False), use_container_width=True)
    else:
        st.warning("현재 구간 데이터가 없습니다.")

    section("리스크 기사 근거")
    article_table(active_df[active_df["sentiment_group"] == "부정/리스크"], DATE_COL, n=100)

# =========================================================
# 5. Co-occurrence
# =========================================================

with tab_network:
    st.subheader("Co-occurrence Analysis")
    method_box(
        "Co-occurrence Analysis",
        "같은 기사 안에 함께 등장한 키워드 쌍을 계산하여 IT 이슈 간 관계를 분석합니다.",
        "기사별 등장 키워드를 추출한 뒤 가능한 키워드 조합의 동시출현 빈도를 계산합니다.",
        "동시출현 빈도가 높은 키워드 쌍은 같은 맥락에서 자주 보도되는 연결 이슈로 해석합니다."
    )

    if net_df.empty:
        st.warning("현재 구간에서 동시출현 네트워크를 만들 수 있는 데이터가 부족합니다.")
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
        st.dataframe(net_df.head(50), use_container_width=True)

        insight_box("해석", [
            f"가장 강한 연결은 '{top_pair['keyword_a']} ↔ {top_pair['keyword_b']}'입니다.",
            "이는 두 키워드가 같은 기사 안에서 자주 함께 등장했다는 의미입니다.",
            "개별 키워드 빈도만으로는 확인하기 어려운 이슈 간 관계 구조를 확인할 수 있습니다."
        ])

# =========================================================
# 6. LDA
# =========================================================

with tab_lda:
    st.subheader("LDA Topic Modeling")
    method_box(
        "LDA Topic Modeling",
        "뉴스 데이터 내부에 숨어 있는 잠재 토픽을 비지도 학습 방식으로 탐색합니다.",
        "CountVectorizer로 문서-단어 행렬을 만들고 Latent Dirichlet Allocation을 적용합니다.",
        "각 토픽의 대표 단어와 토픽별 기사 비중을 확인합니다."
    )

    st.info(
        "LDA는 데이터 수와 기간이 충분할수록 안정적인 결과를 냅니다. "
        "현재 결과는 잠재 토픽 탐색용이며, 향후 수개월 이상의 데이터가 누적되면 더 신뢰도 높은 토픽 분석이 가능합니다."
    )

    lda_topic_count = st.slider("LDA 토픽 수", min_value=3, max_value=8, value=5)
    lda_word_count = st.slider("토픽별 대표 단어 수", min_value=5, max_value=12, value=8)

    lda_topic_df, lda_doc_topics = lda_topic_modeling(
        lda_docs,
        n_topics=lda_topic_count,
        n_words=lda_word_count,
        max_features=1200
    )

    if lda_topic_df.empty:
        st.warning("LDA 분석을 수행하기에 문서 수 또는 단어 수가 부족합니다.")
    else:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.dataframe(lda_topic_df, use_container_width=True)
        with c2:
            progress_list(lda_topic_df.sort_values("article_count", ascending=False), "topic_id", "article_count", "LDA 토픽별 기사 수", top_n=lda_topic_count)

        insight_box("LDA 해석 기준", [
            "각 Topic의 대표 단어를 보고 사람이 토픽명을 해석해야 합니다.",
            "예: 'AI, GPT, OpenAI, GPU'가 함께 나타나면 AI/인공지능 토픽으로 해석할 수 있습니다.",
            "현재 수집 기간이 짧기 때문에 LDA 결과는 확정적인 결론보다 탐색적 분석 결과로 제시하는 것이 적절합니다.",
            "향후 데이터 기간을 확대하면 토픽 안정성과 장기 트렌드 분석을 함께 검증할 수 있습니다."
        ])

# =========================================================
# 7. Evidence
# =========================================================

with tab_evidence:
    st.subheader("Evidence Center")
    st.caption("분석 결과의 근거가 되는 실제 기사 데이터를 확인하는 탭입니다.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        topic_filter = st.selectbox("토픽 필터", ["전체"] + sorted(df["primary_topic"].dropna().unique().tolist()))
    with col2:
        sentiment_filter = st.selectbox("감성 필터", ["전체"] + sorted(df["sentiment_group"].dropna().unique().tolist()))
    with col3:
        source_filter = st.selectbox("소스 필터", ["전체"] + sorted(df["analysis_source"].dropna().unique().tolist()))
    with col4:
        keyword_filter = st.text_input("키워드 검색", "")

    view = df.copy()

    if topic_filter != "전체":
        view = view[view["primary_topic"] == topic_filter]
    if sentiment_filter != "전체":
        view = view[view["sentiment_group"] == sentiment_filter]
    if source_filter != "전체":
        view = view[view["analysis_source"] == source_filter]
    if keyword_filter.strip():
        view = filter_keyword(view, keyword_filter.strip())

    st.info(f"필터 적용 결과: {len(view):,}건")
    article_table(view, DATE_COL, n=300)
