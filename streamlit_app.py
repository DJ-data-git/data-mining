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
st.caption("뉴스 데이터를 단순 나열하지 않고, 분석 질문별로 데이터마이닝 기법을 적용해 발견점과 해석을 도출합니다.")

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

POSITIVE_WORDS = ["성장", "확대", "출시", "투자", "협력", "개선", "강화", "수주", "증가", "성공", "최초", "고도화", "혁신", "개발", "도입", "선정", "공급", "상승", "기대"]
NEGATIVE_WORDS = ["해킹", "침해", "유출", "장애", "중단", "규제", "감소", "적자", "위험", "논란", "피해", "취약점", "공격", "랜섬웨어", "오류", "실패", "제재", "우려", "하락"]
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


def analysis_header(question, method, why, output):
    st.markdown(f"""
    <div style="background:rgba(2,6,23,.76);border:1px solid rgba(56,189,248,.28);
    border-radius:20px;padding:20px;margin-bottom:14px;">
      <div style="font-size:22px;font-weight:950;color:#f8fafc;margin-bottom:10px;">분석 질문: {question}</div>
      <div style="color:#dbeafe;line-height:1.75;"><b>사용 기법</b>: {method}</div>
      <div style="color:#cbd5e1;line-height:1.75;"><b>분석 이유</b>: {why}</div>
      <div style="color:#e0f2fe;line-height:1.75;"><b>확인할 결과</b>: {output}</div>
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


def insight_box(title, bullets, color="#38bdf8"):
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


def safe_top(df, col, default="-"):
    if df is None or df.empty or col not in df.columns:
        return default
    return df.iloc[0][col]


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

# =========================================================
# Sidebar Global Filter
# =========================================================

st.sidebar.header("분석 기준 설정")

period_option = st.sidebar.selectbox(
    "분석 단위 선택",
    ["일별", "주별", "월별", "연별"],
    index=0,
    help="선택한 단위에 따라 Summary와 분석 결과의 기준 구간이 바뀝니다."
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

# Core metrics
active_keyword_count = keyword_counts(active_df, CORE_KEYWORDS)
overall_keyword_count = keyword_counts(df, CORE_KEYWORDS)
active_tfidf = tfidf_keywords(active_df, top_n=20)
overall_tfidf = tfidf_keywords(df, top_n=25)
candidate_tfidf = keyword_candidate_tfidf(active_df, CORE_KEYWORDS, top_n=20)
active_topics = topic_counts(active_df)
keyword_change_df = period_keyword_change(active_df, previous_df, CORE_KEYWORDS)

network_keywords = candidate_tfidf["keyword"].head(25).tolist() if not candidate_tfidf.empty else overall_keyword_count["keyword"].head(25).tolist()
net_df = keyword_network(active_df if len(active_df) else df, network_keywords)

lda_docs = text_series(active_df if len(active_df) >= 20 else df).tolist()
lda_topic_df, lda_doc_topics = lda_topic_modeling(lda_docs, n_topics=5, n_words=8)

# Summary derived findings
top_frequency_keyword = safe_top(active_keyword_count, "keyword")
top_frequency_count = int(safe_top(active_keyword_count, "count", 0)) if not active_keyword_count.empty else 0

top_tfidf_keyword = safe_top(candidate_tfidf, "keyword")
top_tfidf_score = safe_top(candidate_tfidf, "score", 0)

top_change_keyword = safe_top(keyword_change_df, "keyword")
top_change_value = int(safe_top(keyword_change_df, "change", 0)) if not keyword_change_df.empty else 0

top_topic = safe_top(active_topics, "topic")
top_topic_ratio = safe_top(active_topics, "ratio", 0)

risk_df = filter_keywords(active_df, RISK_KEYWORDS)
risk_ratio = round(len(risk_df) / len(active_df) * 100, 1) if len(active_df) else 0

sentiment_df = active_df["sentiment_group"].value_counts().reset_index()
sentiment_df.columns = ["sentiment", "count"] if not sentiment_df.empty else ["sentiment", "count"]
top_sentiment = safe_top(sentiment_df, "sentiment")

top_pair = f"{net_df.iloc[0]['keyword_a']} ↔ {net_df.iloc[0]['keyword_b']}" if not net_df.empty else "-"
top_pair_count = int(net_df.iloc[0]["co_count"]) if not net_df.empty else 0

top_lda_topic = safe_top(lda_topic_df.sort_values("article_count", ascending=False) if not lda_topic_df.empty else lda_topic_df, "topic_id")
top_lda_words = safe_top(lda_topic_df.sort_values("article_count", ascending=False) if not lda_topic_df.empty else lda_topic_df, "top_words")

# =========================================================
# Tabs - Question Driven
# =========================================================

tab_summary, tab_importance, tab_trend, tab_sentiment, tab_network, tab_lda, tab_evidence = st.tabs([
    "1. 핵심 발견 요약",
    "2. 중요한 키워드",
    "3. 급부상 이슈",
    "4. 뉴스 성향",
    "5. 함께 움직이는 기술",
    "6. 숨겨진 토픽",
    "7. 근거 기사"
])

# =========================================================
# 1. Summary
# =========================================================

with tab_summary:
    analysis_header(
        "현재 IT 뉴스에서 무엇을 발견했는가?",
        "Descriptive Analytics + TF-IDF + Time Series + Sentiment + Co-occurrence + LDA",
        "각 분석기법의 결과를 따로 나열하지 않고, 현재 구간에서 도출된 핵심 발견을 한 화면에 요약합니다.",
        "핵심 키워드, 특징 키워드, 급부상 키워드, 리스크 비중, 연결 관계, 잠재 토픽"
    )

    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("현재 분석 구간", current_period_label, f"원본 범위 {min_date} ~ {max_date}")
    with c2:
        card("현재 구간 기사 수", f"{len(active_df):,}건", f"전체 수집 기사 {len(df):,}건")
    with c3:
        card("최다 언급 키워드", top_frequency_keyword, f"{top_frequency_count:,}건 언급")
    with c4:
        card("리스크 기사 비중", f"{risk_ratio}%", f"리스크 기사 {len(risk_df):,}건", "#ef4444")

    section("분석을 통해 발견한 핵심 내용")
    insight_box("핵심 발견", [
        f"단순 빈도 기준으로는 '{top_frequency_keyword}'가 가장 많이 언급되었습니다.",
        f"TF-IDF 기준으로는 '{top_tfidf_keyword}'가 현재 구간을 대표하는 특징 키워드로 나타났습니다.",
        f"직전 구간 대비 가장 크게 증가한 키워드는 '{top_change_keyword}'이며 변화량은 {top_change_value:+,}건입니다.",
        f"주제 분류 기준 가장 큰 비중은 '{top_topic}'이며 전체의 {top_topic_ratio}%입니다.",
        f"감성 분석 기준 현재 구간의 대표 성향은 '{top_sentiment}'입니다.",
        f"동시출현 분석 기준 가장 강한 연결 관계는 '{top_pair}'이며 {top_pair_count:,}건 함께 등장했습니다.",
        f"LDA 분석에서 가장 큰 잠재 토픽은 '{top_lda_topic}'이며 대표 단어는 '{top_lda_words}'입니다."
    ])

    section("현재 구간 주요 데이터")
    col_a, col_b = st.columns(2)
    with col_a:
        progress_list(active_keyword_count.head(10), "keyword", "count", "빈도 기준 핵심 키워드 TOP 10")
    with col_b:
        progress_list(active_topics.head(10), "topic", "count", "주제 분류 결과")

# =========================================================
# 2. Keyword Importance
# =========================================================

with tab_importance:
    analysis_header(
        "많이 나온 키워드와 실제로 중요한 키워드는 같은가?",
        "Keyword Frequency Analysis + TF-IDF Analysis",
        "단순 빈도만 보면 일반적으로 자주 등장하는 단어가 중요해 보일 수 있으므로, TF-IDF로 현재 구간을 더 잘 대표하는 키워드를 따로 확인합니다.",
        "빈도 TOP 키워드와 TF-IDF TOP 키워드를 비교하여 '많이 언급된 이슈'와 '특징적인 이슈'를 구분합니다."
    )

    freq_top = active_keyword_count.head(10).copy()
    tfidf_top = candidate_tfidf.head(10).copy()

    f1, f2, f3 = st.columns(3)
    with f1:
        card("빈도 1위", top_frequency_keyword, f"{top_frequency_count:,}건 등장")
    with f2:
        card("TF-IDF 1위", top_tfidf_keyword, f"점수 {top_tfidf_score}")
    with f3:
        same_or_diff = "같음" if top_frequency_keyword == top_tfidf_keyword else "다름"
        card("빈도 vs TF-IDF", same_or_diff, "두 결과가 다르면 특징 키워드가 따로 존재한다는 의미")

    section("빈도 분석과 TF-IDF 분석 비교")
    col1, col2 = st.columns(2)
    with col1:
        progress_list(freq_top, "keyword", "count", "단순 빈도 TOP 10")
        st.dataframe(freq_top, use_container_width=True)
    with col2:
        progress_list(tfidf_top, "keyword", "score", "TF-IDF 중요도 TOP 10", suffix="점")
        st.dataframe(tfidf_top, use_container_width=True)

    common = set(freq_top["keyword"]).intersection(set(tfidf_top["keyword"])) if not freq_top.empty and not tfidf_top.empty else set()
    only_tfidf = [kw for kw in tfidf_top["keyword"].tolist() if kw not in freq_top["keyword"].tolist()] if not tfidf_top.empty and not freq_top.empty else []

    insight_box("분석 해석", [
        f"빈도 1위는 '{top_frequency_keyword}', TF-IDF 1위는 '{top_tfidf_keyword}'입니다.",
        f"빈도 TOP 10과 TF-IDF TOP 10의 공통 키워드는 {len(common)}개입니다.",
        f"TF-IDF에만 강하게 나타난 키워드는 {', '.join(only_tfidf[:5]) if only_tfidf else '없음'}입니다.",
        "따라서 이 화면에서는 단순히 많이 나온 키워드가 아니라, 현재 뉴스 구간을 특징짓는 키워드를 확인할 수 있습니다."
    ])

    if not candidate_tfidf.empty:
        selected_kw = st.selectbox("TF-IDF 키워드 근거 기사 확인", candidate_tfidf["keyword"].tolist())
        selected_df = filter_keyword(active_df, selected_kw)
        st.info(f"'{selected_kw}' 관련 기사: {len(selected_df):,}건")
        article_table(selected_df, DATE_COL, n=100)

# =========================================================
# 3. Trend
# =========================================================

with tab_trend:
    analysis_header(
        "최근 어떤 이슈가 급부상했는가?",
        "Time Series Analysis + Period-over-Period Change Analysis",
        "일/주/월/년 단위로 데이터를 재집계하고 직전 구간과 비교해 증가한 키워드를 탐지합니다.",
        "현재 구간에서 새롭게 증가한 키워드와 기사량이 집중된 구간을 확인합니다."
    )

    period_count_df = period_counts(summary_df, PERIOD_COL, period_label_map)
    peak_period = period_count_df.sort_values("article_count", ascending=False).iloc[0]["period_label"] if not period_count_df.empty else "-"
    peak_count = int(period_count_df.sort_values("article_count", ascending=False).iloc[0]["article_count"]) if not period_count_df.empty else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        card("현재 분석 구간", current_period_label, period_option)
    with c2:
        card("기사량 최다 구간", peak_period, f"{peak_count:,}건")
    with c3:
        card("최대 증가 키워드", top_change_keyword, f"{top_change_value:+,}건")

    section("기간별 기사량")
    col1, col2 = st.columns(2)
    with col1:
        progress_list(period_count_df.sort_values("article_count", ascending=False), "period_label", "article_count", f"{period_option} 기사량 TOP 10")
    with col2:
        st.dataframe(period_count_df, use_container_width=True)

    section("직전 구간 대비 증가 키워드")
    st.dataframe(keyword_change_df, use_container_width=True)
    progress_list(keyword_change_df.head(10), "keyword", "change", "증가 키워드 TOP 10", suffix="건")

    insight_box("분석 해석", [
        f"현재 분석 단위는 '{period_option}'이며 현재 구간은 '{current_period_label}'입니다.",
        f"기사량이 가장 많았던 구간은 '{peak_period}'이며 {peak_count:,}건이 수집되었습니다.",
        f"직전 구간 대비 가장 크게 증가한 키워드는 '{top_change_keyword}'입니다.",
        "이 결과는 특정 이슈가 단기적으로 부상했는지 확인하는 데 사용됩니다."
    ])

# =========================================================
# 4. Sentiment
# =========================================================

with tab_sentiment:
    analysis_header(
        "IT 뉴스는 성장 이슈 중심인가, 리스크 이슈 중심인가?",
        "Rule-based Sentiment Analysis + Risk Keyword Analysis",
        "뉴스 제목과 요약문에서 긍정/성장 키워드와 부정/리스크 키워드를 비교해 기사 성향을 분류합니다.",
        "긍정/중립/부정 비중과 리스크가 높은 주제를 확인합니다."
    )

    sentiment_count = active_df["sentiment_group"].value_counts().reset_index()
    sentiment_count.columns = ["sentiment", "count"] if not sentiment_count.empty else ["sentiment", "count"]
    if not sentiment_count.empty:
        sentiment_count["ratio"] = (sentiment_count["count"] / len(active_df) * 100).round(1)

    neg_count = int(sentiment_count[sentiment_count["sentiment"] == "부정/리스크"]["count"].sum()) if not sentiment_count.empty else 0
    pos_count = int(sentiment_count[sentiment_count["sentiment"] == "긍정/성장"]["count"].sum()) if not sentiment_count.empty else 0
    neg_ratio = round(neg_count / len(active_df) * 100, 1) if len(active_df) else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        card("대표 성향", top_sentiment, "현재 구간 기준")
    with c2:
        card("긍정/성장 기사", f"{pos_count:,}건", "투자·출시·성장 등")
    with c3:
        card("부정/리스크 기사", f"{neg_count:,}건", f"전체 대비 {neg_ratio}%", "#ef4444")

    section("감성 분포")
    col1, col2 = st.columns(2)
    with col1:
        progress_list(sentiment_count, "sentiment", "count", "현재 구간 감성 분포")
    with col2:
        st.dataframe(sentiment_count, use_container_width=True)

    section("주제별 리스크 비율")
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
        topic_sentiment = topic_sentiment.sort_values("risk_ratio", ascending=False)
        st.dataframe(topic_sentiment, use_container_width=True)
        top_risk_topic = topic_sentiment.iloc[0]["primary_topic"] if not topic_sentiment.empty else "-"
        top_risk_ratio = topic_sentiment.iloc[0]["risk_ratio"] if not topic_sentiment.empty else 0
    else:
        top_risk_topic = "-"
        top_risk_ratio = 0
        st.warning("현재 구간 데이터가 없습니다.")

    insight_box("분석 해석", [
        f"현재 구간의 대표 감성은 '{top_sentiment}'입니다.",
        f"부정/리스크 기사는 {neg_count:,}건이며 전체 대비 {neg_ratio}%입니다.",
        f"리스크 비율이 가장 높은 주제는 '{top_risk_topic}'이며 리스크 비율은 {top_risk_ratio}%입니다.",
        "이를 통해 단순 기술 성장 뉴스뿐 아니라 보안·개인정보·장애 등 위험 신호도 함께 확인할 수 있습니다."
    ])

# =========================================================
# 5. Co-occurrence
# =========================================================

with tab_network:
    analysis_header(
        "어떤 기술 이슈들이 함께 움직이는가?",
        "Co-occurrence Analysis + Network Analysis",
        "같은 기사 안에 함께 등장한 키워드 쌍을 계산해 이슈 간 관계를 확인합니다.",
        "강한 연결 관계를 통해 AI-반도체, 클라우드-데이터센터 같은 연결 구조를 파악합니다."
    )

    if net_df.empty:
        st.warning("현재 구간에서 동시출현 네트워크를 만들 수 있는 데이터가 부족합니다.")
    else:
        top_pair_row = net_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            card("최강 연결", f"{top_pair_row['keyword_a']} ↔ {top_pair_row['keyword_b']}", f"{int(top_pair_row['co_count']):,}건 동시출현")
        with c2:
            card("관계 수", f"{len(net_df):,}개", "키워드 페어 기준")
        with c3:
            nodes = len(set(net_df["keyword_a"]).union(set(net_df["keyword_b"])))
            card("연결 키워드 수", f"{nodes:,}개", "네트워크 노드 기준")

        components.html(network_html(net_df), height=720, scrolling=True)
        st.dataframe(net_df.head(50), use_container_width=True)

        insight_box("분석 해석", [
            f"가장 강한 연결은 '{top_pair}'입니다.",
            f"해당 키워드 조합은 {top_pair_count:,}건의 기사에서 함께 등장했습니다.",
            "이는 두 이슈가 같은 맥락에서 자주 보도되었다는 의미입니다.",
            "개별 키워드 빈도만으로는 확인하기 어려운 이슈 간 연결 구조를 확인할 수 있습니다."
        ])

# =========================================================
# 6. LDA
# =========================================================

with tab_lda:
    analysis_header(
        "뉴스 속 숨겨진 주제는 무엇인가?",
        "LDA Topic Modeling",
        "사전에 정한 분류표가 아니라, 뉴스 데이터 내부에서 함께 등장하는 단어 패턴을 바탕으로 잠재 토픽을 탐색합니다.",
        "각 토픽의 대표 단어와 기사 비중을 통해 숨겨진 뉴스 주제를 해석합니다."
    )

    st.info(
        "LDA는 데이터 수와 기간이 충분할수록 안정적인 결과를 냅니다. "
        "현재 결과는 잠재 토픽 탐색 결과로 제시하고, 향후 데이터가 누적되면 더 신뢰도 높은 토픽 분석으로 확장할 수 있습니다."
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
        lda_topic_df = lda_topic_df.sort_values("article_count", ascending=False).reset_index(drop=True)
        top_lda = lda_topic_df.iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            card("최대 토픽", top_lda["topic_id"], f"{int(top_lda['article_count']):,}건")
        with c2:
            card("대표 단어", top_lda["top_words"].split(",")[0], top_lda["top_words"])
        with c3:
            card("토픽 비중", f"{top_lda['ratio']}%", "LDA 기준")

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.dataframe(lda_topic_df, use_container_width=True)
        with col2:
            progress_list(lda_topic_df, "topic_id", "article_count", "LDA 토픽별 기사 수", top_n=lda_topic_count)

        insight_box("분석 해석", [
            f"가장 큰 잠재 토픽은 '{top_lda['topic_id']}'이며 전체의 {top_lda['ratio']}%를 차지합니다.",
            f"해당 토픽의 대표 단어는 '{top_lda['top_words']}'입니다.",
            "대표 단어 조합을 바탕으로 사람이 토픽명을 해석해야 합니다.",
            "현재 데이터 기간이 짧기 때문에 LDA 결과는 확정적 결론보다 탐색적 분석 결과로 제시하는 것이 적절합니다."
        ])

# =========================================================
# 7. Evidence
# =========================================================

with tab_evidence:
    analysis_header(
        "분석 결과의 근거 기사는 무엇인가?",
        "Evidence-based Validation",
        "분석 결과가 단순 수치에 그치지 않도록 실제 기사 제목, 요약, 링크를 확인합니다.",
        "키워드, 주제, 감성, 소스별로 분석 근거 기사를 검증합니다."
    )

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
