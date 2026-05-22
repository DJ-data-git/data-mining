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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# Page / Style
# =========================================================

st.set_page_config(page_title="IT News Intelligence Dashboard", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a 0%,#111827 45%,#020617 100%); color:#e5e7eb;}
.block-container {padding-top:2rem; padding-bottom:3rem;}
h1 {color:#e5e7eb; font-weight:900; letter-spacing:-0.05em;}
h2,h3 {color:#e5e7eb; font-weight:800;}
[data-testid="stCaptionContainer"] {color:#94a3b8;}
[data-testid="stDataFrame"] {border-radius:16px; overflow:hidden; border:1px solid rgba(148,163,184,.18);}
.stAlert {background:rgba(14,165,233,.12); border:1px solid rgba(56,189,248,.35); border-radius:16px; color:#e0f2fe;}
.stButton > button {background:linear-gradient(135deg,#0284c7,#2563eb); color:white; border:0; border-radius:999px; padding:.5rem 1rem; font-weight:700;}
.stButton > button:hover {background:linear-gradient(135deg,#38bdf8,#2563eb); color:white; border:0;}
.stTextInput input {background:#020617; color:#e5e7eb; border:1px solid #334155; border-radius:12px;}
hr {border:none; height:1px; background:linear-gradient(90deg,transparent,#38bdf8,transparent); margin:2rem 0;}
section[data-testid="stSidebar"] {background:#020617;}
</style>
""", unsafe_allow_html=True)

st.title("Today’s IT Trend Radar")

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

MAIN_KEYWORDS = [
    "AI", "인공지능", "생성형AI", "챗GPT", "반도체", "클라우드", "보안", "데이터",
    "로봇", "배터리", "전기차", "삼성", "네이버", "카카오", "엔비디아",
    "AWS", "Azure", "해킹", "개인정보", "데이터센터", "HBM"
]

TOPIC_MAP = {
    "AI/인공지능": ["AI", "인공지능", "생성형AI", "챗GPT", "엔비디아"],
    "반도체": ["반도체", "삼성전자", "SK하이닉스", "칩", "HBM"],
    "클라우드/데이터센터": ["클라우드", "AWS", "Azure", "데이터센터"],
    "보안/개인정보": ["보안", "해킹", "개인정보", "랜섬웨어", "침해"],
    "모빌리티/로봇": ["전기차", "자율주행", "배터리", "로봇"],
    "플랫폼/빅테크": ["네이버", "카카오", "구글", "애플", "메타"],
}

IT_TFIDF_CANDIDATES = [
    "AI", "인공지능", "생성형AI", "생성형 AI", "LLM", "챗GPT", "ChatGPT", "GPT", "OpenAI",
    "멀티모달", "AI 에이전트", "AI 검색", "AI 서비스",
    "반도체", "AI반도체", "AI 반도체", "HBM", "GPU", "엔비디아", "NVIDIA",
    "삼성전자", "SK하이닉스", "파운드리", "메모리", "칩", "첨단 반도체",
    "클라우드", "AWS", "Azure", "데이터센터", "데이터 센터", "서버", "인프라",
    "SaaS", "쿠버네티스", "Kubernetes", "클라우드 전환",
    "보안", "해킹", "개인정보", "랜섬웨어", "침해", "취약점", "사이버 공격",
    "정보보호", "제로트러스트", "인증", "망분리",
    "로봇", "전기차", "배터리", "자율주행", "모빌리티", "이차전지", "전장",
    "네이버", "카카오", "구글", "애플", "메타", "마이크로소프트", "MS",
    "플랫폼", "빅테크", "검색", "커머스",
    "데이터", "빅데이터", "소프트웨어", "디지털", "DX", "디지털전환",
    "핀테크", "블록체인", "가상자산", "메타버스"
]

STOPWORDS = [
    "the", "and", "for", "that", "with", "this", "from", "have", "will", "into", "about",
    "their", "they", "them", "were", "been", "being", "said", "more", "than", "over",
    "after", "before", "while", "where", "when", "what", "which", "would", "could", "should",
    "there", "these", "those", "because", "through", "during", "under", "between", "among",
    "it", "to", "of", "in", "is", "on", "at", "by", "be", "as", "an", "or", "if", "we",
    "he", "she", "you", "are", "was", "has", "had", "can", "not", "also", "but", "how",
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
    border-radius:22px;padding:22px;min-height:145px;box-shadow:0 0 24px rgba(56,189,248,.08);">
      <div style="color:#94a3b8;font-size:14px;font-weight:700;margin-bottom:8px;">{title}</div>
      <div style="color:{color};font-size:34px;font-weight:900;margin-bottom:8px;">{value}</div>
      <div style="color:#cbd5e1;font-size:14px;line-height:1.6;">{desc}</div>
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
    colors = ["#38bdf8", "#60a5fa", "#818cf8", "#22c55e", "#14b8a6", "#eab308", "#f97316", "#ef4444", "#ec4899", "#a855f7"]

    for idx, (_, row) in enumerate(view.iterrows()):
        label, value = row[label_col], float(row[value_col])
        ratio = value / max_v * 100 if max_v else 0
        color = colors[idx % len(colors)]
        value_text = f"{int(value):,}{suffix}" if value == int(value) else f"{value:.2f}"
        st.markdown(f"""
        <div style="background:rgba(15,23,42,.75);border:1px solid rgba(148,163,184,.12);border-radius:18px;
        padding:14px 18px;margin-bottom:10px;box-shadow:0 0 14px rgba(56,189,248,.04);">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div style="font-size:17px;font-weight:850;color:white;">{label}</div>
            <div style="font-size:16px;font-weight:850;color:{color};">{value_text}</div>
          </div>
          <div style="width:100%;height:9px;background:#1e293b;border-radius:999px;overflow:hidden;">
            <div style="width:{ratio}%;height:100%;background:linear-gradient(90deg,{color},#38bdf8);border-radius:999px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


def keyword_chip_grid(df, label_col="keyword", value_col="count", title=None, top_n=10, suffix="건", clickable=False, session_key="drill_keyword"):
    if title:
        st.markdown(f"### {title}")
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        st.warning("표시할 데이터가 없습니다.")
        return

    view = df.head(top_n).copy().reset_index(drop=True)
    max_v = view[value_col].max()
    colors = ["#38bdf8", "#60a5fa", "#818cf8", "#22c55e", "#14b8a6", "#eab308", "#f97316", "#ef4444", "#ec4899", "#a855f7"]

    for start in range(0, len(view), 5):
        cols = st.columns(5)
        for offset, (col, (_, row)) in enumerate(zip(cols, view.iloc[start:start+5].iterrows())):
            idx = start + offset
            label = str(row[label_col])
            value = float(row[value_col])
            ratio = value / max_v * 100 if max_v else 0
            color = colors[idx % len(colors)]
            value_text = f"{int(value):,}{suffix}" if value == int(value) else f"{value:.2f}"
            with col:
                st.markdown(f"""
                <div style="background:rgba(15,23,42,.82);border:1px solid rgba(56,189,248,.22);border-radius:18px;
                padding:16px 14px;margin-bottom:8px;min-height:112px;box-shadow:0 0 16px rgba(56,189,248,.05);">
                    <div style="font-size:18px;font-weight:900;color:#e5e7eb;margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        {label}
                    </div>
                    <div style="font-size:24px;font-weight:950;color:{color};margin-bottom:10px;">
                        {value_text}
                    </div>
                    <div style="width:100%;height:7px;background:#1e293b;border-radius:999px;overflow:hidden;">
                        <div style="width:{ratio}%;height:100%;background:linear-gradient(90deg,{color},#38bdf8);border-radius:999px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if clickable:
                    current_selected = st.session_state.get(session_key, "")
                    button_label = "뉴스 닫기" if current_selected == label else "뉴스 보기"

                    if st.button(button_label, key=f"{session_key}_{idx}_{label}"):
                        if st.session_state.get(session_key, "") == label:
                            st.session_state[session_key] = ""
                        else:
                            st.session_state[session_key] = label


def methodology_cards():
    methods = [
        ("TF-IDF", "일반 단어 전체가 아니라 IT 키워드 후보군 안에서 중요도를 계산합니다."),
        ("Cosine Similarity", "TF-IDF 벡터 간 각도 유사도를 계산해 특정 키워드와 유사한 기사를 찾습니다."),
        ("Co-occurrence", "같은 기사 안에 함께 등장한 키워드를 분석해 이슈 간 연결 구조를 파악합니다."),
        ("Sentiment", "성장·투자·혁신 또는 해킹·유출·침해 키워드로 보도 성향을 분류합니다."),
        ("Time-Series", "날짜별 기사량 변화를 추적해 특정 이슈가 언제 집중되었는지 확인합니다."),
    ]
    cols = st.columns(5)
    for col, (title, desc) in zip(cols, methods):
        with col:
            card(title, "", desc, "#38bdf8")

# =========================================================
# Data Utilities
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
    date_col = next((c for c in ["pubDate_dt", "pubDate_ymd", "pubDate", "date", "published_date"] if c in df.columns), None)
    if not date_col:
        df["analysis_date"] = "unknown"
        return df, "analysis_date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["analysis_date"] = df[date_col].dt.strftime("%Y-%m-%d").fillna("unknown")
    return df, "analysis_date"


def prepare_df(df):
    df, date_col = normalize_date(df)
    for col in ["source", "source_group", "title", "description", "originallink", "link"]:
        if col not in df.columns:
            df[col] = ""

    df["source"] = df["source"].fillna("").astype(str)
    df.loc[df["source"].str.strip() == "", "source"] = "unknown"

    df["source_group"] = df["source_group"].fillna("").astype(str)
    empty_group = df["source_group"].str.strip() == ""
    df.loc[empty_group, "source_group"] = df.loc[empty_group, "source"]

    df["analysis_source"] = df["source"].fillna("unknown").astype(str)
    df.loc[df["analysis_source"].str.strip() == "", "analysis_source"] = "unknown"

    dedup_cols = [c for c in ["originallink", "link", "title"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last")
    return df, date_col


def text_series(df):
    return df["title"].fillna("").astype(str) + " " + df["description"].fillna("").astype(str)


def filter_keyword(df, keyword):
    return df[text_series(df).str.contains(keyword, case=False, regex=False)]


def filter_keywords(df, keywords):
    cond = pd.Series(False, index=df.index)
    txt = text_series(df)
    for kw in keywords:
        cond |= txt.str.contains(kw, case=False, regex=False)
    return df[cond]


def keyword_counts(df, keywords):
    txt = text_series(df)
    return pd.DataFrame([
        {"keyword": kw, "count": int(txt.str.contains(kw, case=False, regex=False).sum())}
        for kw in keywords
    ]).sort_values("count", ascending=False)

# =========================================================
# Analysis Functions
# =========================================================

def classify_sentiment(df):
    pos_words = ["성장", "확대", "출시", "투자", "협력", "개선", "강화", "수주", "증가", "성공", "최초", "고도화", "혁신"]
    neg_words = ["해킹", "침해", "유출", "장애", "중단", "규제", "감소", "적자", "위험", "논란", "피해", "취약점", "공격"]
    labels = []
    for text in text_series(df):
        pos, neg = sum(w in text for w in pos_words), sum(w in text for w in neg_words)
        labels.append("긍정/성장" if pos > neg else "부정/리스크" if neg > pos else "중립")
    return labels


def topic_counts(df):
    return pd.DataFrame([
        {"topic": topic, "count": len(filter_keywords(df, kws)), "keywords": ", ".join(kws)}
        for topic, kws in TOPIC_MAP.items()
    ]).sort_values("count", ascending=False)


def topic_sentiment(df):
    rows = []
    for topic, kws in TOPIC_MAP.items():
        sub = filter_keywords(df, kws)
        total = len(sub)
        counts = sub["sentiment_group"].value_counts() if total else pd.Series(dtype=int)
        pos, neu, risk = int(counts.get("긍정/성장", 0)), int(counts.get("중립", 0)), int(counts.get("부정/리스크", 0))
        rows.append({
            "topic": topic, "total": total, "긍정/성장": pos, "중립": neu, "부정/리스크": risk,
            "positive_ratio": round(pos / total * 100, 1) if total else 0,
            "risk_ratio": round(risk / total * 100, 1) if total else 0,
        })
    return pd.DataFrame(rows).sort_values("total", ascending=False)


def daily_top_keywords(df, date_col, top_n=5):
    rows = []
    for date in sorted(df[date_col].dropna().unique()):
        sub = df[df[date_col] == date]
        for rank, row in enumerate(keyword_counts(sub, MAIN_KEYWORDS).head(top_n).itertuples(), 1):
            rows.append({"date": date, "rank": rank, "keyword": row.keyword, "count": row.count})
    return pd.DataFrame(rows)


def topic_timeseries(df, date_col):
    rows = []
    for date in sorted(df[date_col].dropna().unique()):
        sub = df[df[date_col] == date]
        row = {"date": date}
        for topic, kws in TOPIC_MAP.items():
            row[topic] = len(filter_keywords(sub, kws))
        rows.append(row)
    return pd.DataFrame(rows)


def keyword_network(df):
    rows = []
    for text in text_series(df):
        appeared = sorted({kw for kw in MAIN_KEYWORDS if kw.lower() in text.lower()})
        rows.extend(combinations(appeared, 2))
    if not rows:
        return pd.DataFrame(columns=["keyword_a", "keyword_b", "co_count"])
    return pd.DataFrame(rows, columns=["keyword_a", "keyword_b"]).value_counts().reset_index(name="co_count").sort_values("co_count", ascending=False)


def tfidf_keywords(df, top_n=20):
    """IT 키워드 사전 기반 TF-IDF 유사 중요도."""
    txt = text_series(df).fillna("").astype(str)
    if len(txt) == 0 or txt.str.strip().eq("").all():
        return pd.DataFrame(columns=["keyword", "score", "article_count"])

    total_docs = len(txt)
    rows = []
    for keyword in IT_TFIDF_CANDIDATES:
        contains = txt.str.contains(keyword, case=False, regex=False)
        article_count = int(contains.sum())
        if article_count == 0:
            continue
        idf = 1 + math.log((1 + total_docs) / (1 + article_count))
        score = article_count * idf
        rows.append({"keyword": keyword, "score": round(score, 4), "article_count": article_count})

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["keyword", "score", "article_count"])
    return result.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def similar_articles(df, keyword, top_n=10):
    txt = text_series(df).fillna("").astype(str)
    if len(txt) < 2:
        return pd.DataFrame()
    vec = TfidfVectorizer(max_features=1200, stop_words=STOPWORDS, token_pattern=r"(?u)\b[가-힣A-Za-z0-9]{2,}\b")
    try:
        mat = vec.fit_transform(txt.tolist() + [keyword])
        scores = cosine_similarity(mat[:-1], mat[-1]).flatten()
    except ValueError:
        return pd.DataFrame()
    out = df.copy()
    out["similarity_score"] = scores
    cols = [c for c in ["analysis_date", "analysis_source", "source_group", "title", "description", "originallink", "link", "similarity_score"] if c in out.columns]
    return out.sort_values("similarity_score", ascending=False)[cols].head(top_n)


def source_keyword_table(df):
    rows = []
    for src in df["analysis_source"].value_counts().head(15).index:
        sub = df[df["analysis_source"] == src]
        row = {"source": src, "total_articles": len(sub)}
        for _, r in keyword_counts(sub, MAIN_KEYWORDS).iterrows():
            row[r["keyword"]] = int(r["count"])
        rows.append(row)
    return pd.DataFrame(rows)


def source_topic_ratio(df):
    rows = []
    for src in df["analysis_source"].value_counts().head(15).index:
        sub = df[df["analysis_source"] == src]
        total = len(sub)
        row = {"source": src, "total": total}
        for topic, kws in TOPIC_MAP.items():
            row[topic] = round(len(filter_keywords(sub, kws)) / total * 100, 1) if total else 0
        rows.append(row)
    return pd.DataFrame(rows)


def source_frame(df):
    frames = {
        "성장/혁신": ["성장", "혁신", "출시", "확대", "협력", "투자", "강화"],
        "보안/리스크": ["해킹", "침해", "유출", "장애", "공격", "위험", "취약점"],
        "산업/경쟁": ["시장", "경쟁", "점유율", "HBM", "반도체", "공급망", "수출"],
        "정책/규제": ["정부", "규제", "정책", "법안", "지원", "제도"],
        "인프라/클라우드": ["클라우드", "데이터센터", "AWS", "Azure", "서버", "인프라"],
    }
    rows = []
    for src in df["analysis_source"].value_counts().head(15).index:
        sub = df[df["analysis_source"] == src]
        txt = text_series(sub)
        row = {"source": src, "total_articles": len(sub)}
        for frame, kws in frames.items():
            row[frame] = int(sum(txt.str.contains(kw, case=False, regex=False).sum() for kw in kws))
        rows.append(row)
    return pd.DataFrame(rows)


def event_annotations(df, date_col):
    return pd.DataFrame([
        {"date": d, "event": f"기사 급증 ({int(c):,}건)", "analysis": "기업 발표, 보안 사고, 기술 행사, 정책 발표 등 실제 이벤트와 연결 가능"}
        for d, c in df[date_col].value_counts().sort_values(ascending=False).head(5).items()
    ])


def network_html(net_df, top_n=25):
    G = nx.Graph()
    for _, row in net_df.head(top_n).iterrows():
        G.add_edge(row["keyword_a"], row["keyword_b"], value=int(row["co_count"]), title=f"동시출현: {int(row['co_count'])}")
    net = Network(height="720px", width="100%", bgcolor="#0f172a", font_color="white")
    net.from_nx(G)
    degrees = dict(G.degree())
    for node in net.nodes:
        node["size"] = 18 + degrees.get(node["id"], 1) * 5
        node["color"] = "#38bdf8"
    for edge in net.edges:
        edge["color"] = "#64748b"
        edge["smooth"] = True
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    net.save_graph(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def article_table(df, date_col="analysis_date"):
    cols = [c for c in [date_col, "analysis_source", "source_group", "title", "description", "originallink", "link"] if c in df.columns]
    if date_col in df.columns:
        df = df.sort_values(date_col, ascending=False)
    st.dataframe(df[cols], use_container_width=True)

# =========================================================
# Load + Compute
# =========================================================

keys = [k for k in list_csv_keys(BUCKET_NAME, S3_PREFIX) if (key_date(k) and key_date(k) >= START_DATE)]
if not keys:
    st.error("2026년 5월 14일 이후 CSV 파일을 찾지 못했습니다.")
    st.stop()

df = load_csvs(BUCKET_NAME, sorted(keys))
df, DATE_COL = prepare_df(df)

latest_date = df[DATE_COL].dropna().max()
latest_df = df[df[DATE_COL] == latest_date].copy()

df["sentiment_group"] = classify_sentiment(df)
latest_df["sentiment_group"] = classify_sentiment(latest_df)

top_keywords = keyword_counts(latest_df, MAIN_KEYWORDS).head(10)
daily_kw = daily_top_keywords(df, DATE_COL)
net_df = keyword_network(df)
topic_df = topic_counts(latest_df)
topic_sent_df = topic_sentiment(df)
topic_ts_df = topic_timeseries(df, DATE_COL)

# =========================================================
# Sidebar Menu
# =========================================================

# =========================================================
# Accordion Sidebar Menu
# =========================================================

MENU_TREE = {
    "Home": {
        "type": "single",
        "label": "오늘의 뉴스",
        "desc": "오늘의 핵심 뉴스와 키워드"
    },
    "텍스트 분석": {
        "type": "group",
        "children": {
            "Keyword": "키워드 분석",
            "Trend": "시계열 분석",
            "Network": "네트워크 분석"
        }
    },
    "비교 분석": {
        "type": "group",
        "children": {
            "Source": "소스 분석",
            "Sentiment": "감성/리스크"
        }
    },
    "데이터 탐색": {
        "type": "group",
        "children": {
            "Search": "기사 검색"
        }
    }
}

MENU_LABELS = {
    "Home": "오늘의 뉴스",
    "Keyword": "키워드 분석",
    "Trend": "시계열 분석",
    "Network": "네트워크 분석",
    "Source": "소스 분석",
    "Sentiment": "감성/리스크",
    "Search": "기사 검색",
}

MENU_DESC = {
    "Home": "오늘의 핵심 뉴스와 키워드",
    "Keyword": "TF-IDF · 유사 기사 · 관련 기사",
    "Trend": "날짜별 키워드와 주제 변화",
    "Network": "키워드 동시출현 관계",
    "Source": "수집 소스별 보도 경향",
    "Sentiment": "긍정/리스크 보도 성향",
    "Search": "전체 기사 검색"
}

MENU_ICON = {
    "Home": "H",
    "Keyword": "K",
    "Trend": "T",
    "Network": "N",
    "Source": "S",
    "Sentiment": "R",
    "Search": "Q"
}

if "menu" not in st.session_state:
    st.session_state["menu"] = "Home"
if "open_group" not in st.session_state:
    st.session_state["open_group"] = "텍스트 분석"

st.sidebar.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #030712 55%, #020617 100%);
}
.sidebar-brand {
    padding: 1.1rem 0 1.2rem 0;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid rgba(148,163,184,.14);
}
.sidebar-brand-title {
    font-size: 18px;
    font-weight: 950;
    color: #f8fafc;
    letter-spacing: -0.03em;
}
.sidebar-brand-sub {
    margin-top: 6px;
    color: #64748b;
    font-size: 12px;
    line-height: 1.5;
}
.sidebar-group-card {
    background: rgba(15,23,42,.42);
    border: 1px solid rgba(148,163,184,.13);
    border-radius: 16px;
    padding: 0.72rem 0.78rem;
    margin: 0.48rem 0;
    color: #e5e7eb;
    font-size: 15px;
    font-weight: 950;
}
.sidebar-group-open {
    background: linear-gradient(135deg, rgba(14,165,233,.18), rgba(37,99,235,.10));
    border: 1px solid rgba(56,189,248,.45);
    box-shadow: 0 0 18px rgba(56,189,248,.06);
}
.sidebar-single-current {
    position: relative;
    background: linear-gradient(135deg, rgba(14,165,233,.26), rgba(37,99,235,.18));
    border: 1px solid rgba(56,189,248,.58);
    color: #f8fafc;
    border-radius: 16px;
    padding: 0.82rem 0.82rem;
    margin-bottom: 0.55rem;
    box-shadow: 0 0 22px rgba(56,189,248,.10), inset 0 1px 0 rgba(255,255,255,.05);
}
.sidebar-current {
    background: rgba(56,189,248,.14);
    border: 1px solid rgba(56,189,248,.42);
    color: #f8fafc;
    border-radius: 13px;
    padding: 0.58rem 0.72rem;
    margin: 0.32rem 0 0.32rem 0.55rem;
    font-size: 13px;
    font-weight: 900;
}
.sidebar-row-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 14px;
    font-weight: 850;
}
.sidebar-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 9px;
    background: rgba(56,189,248,.13);
    color: #7dd3fc;
    font-size: 12px;
    font-weight: 950;
}
.sidebar-desc {
    margin-left: 34px;
    margin-top: 5px;
    color: #94a3b8;
    font-size: 11px;
    line-height: 1.4;
}
section[data-testid="stSidebar"] button {
    width: 100%;
    border-radius: 14px !important;
    border: 1px solid rgba(148,163,184,.14) !important;
    background: rgba(15,23,42,.48) !important;
    color: #cbd5e1 !important;
    text-align: left !important;
    padding: 0.56rem 0.72rem !important;
    margin-bottom: 0.38rem !important;
    box-shadow: none !important;
    transition: all .15s ease !important;
    font-size: 13px !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] button:hover {
    transform: translateX(2px);
    background: rgba(30,41,59,.92) !important;
    border: 1px solid rgba(56,189,248,.65) !important;
    box-shadow: 0 0 18px rgba(56,189,248,.08) !important;
}
.child-wrap {
    margin: 0.15rem 0 0.7rem 0.2rem;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">IT News Dashboard</div>
        <div class="sidebar-brand-sub">뉴스 데이터 기반 트렌드 인텔리전스</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Home은 하위 메뉴 없이 단독 메뉴로 처리
if st.session_state["menu"] == "Home":
    st.sidebar.markdown(
        f"""
        <div class="sidebar-single-current">
            <div class="sidebar-row-title">
                <span class="sidebar-icon">{MENU_ICON['Home']}</span>
                <span>오늘의 뉴스</span>
            </div>
            <div class="sidebar-desc">{MENU_DESC['Home']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    if st.sidebar.button("H   오늘의 뉴스", key="menu_Home"):
        st.session_state["menu"] = "Home"
        st.rerun()

for group_name, group_data in MENU_TREE.items():
    if group_data.get("type") != "group":
        continue

    children = group_data["children"]
    is_open = st.session_state.get("open_group") == group_name or st.session_state.get("menu") in children
    arrow = "▾" if is_open else "▸"

    # 상위 메뉴 자체를 클릭하면 열고 닫히도록 처리
    if st.sidebar.button(f"{arrow}  {group_name}", key=f"toggle_{group_name}"):
        st.session_state["open_group"] = "" if is_open else group_name
        st.rerun()

    if is_open:
        st.sidebar.markdown('<div class="child-wrap">', unsafe_allow_html=True)
        for key, label in children.items():
            icon = MENU_ICON.get(key, "•")
            if st.session_state["menu"] == key:
                st.sidebar.markdown(f'<div class="sidebar-current">{icon}  {label}</div>', unsafe_allow_html=True)
            else:
                if st.sidebar.button(f"{icon}   {label}", key=f"menu_{key}"):
                    st.session_state["menu"] = key
                    st.session_state["open_group"] = group_name
                    st.rerun()
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

menu = st.session_state["menu"]

# =========================================================
# Pages
# =========================================================

if menu == "Home":
    st.subheader("Today’s IT News Home")

    top_kw = top_keywords.iloc[0]["keyword"] if not top_keywords.empty else "-"
    top_kw_count = int(top_keywords.iloc[0]["count"]) if not top_keywords.empty else 0

    latest_topic_df = topic_counts(latest_df)
    top_topic = latest_topic_df.iloc[0]["topic"] if not latest_topic_df.empty else "-"
    top_topic_count = int(latest_topic_df.iloc[0]["count"]) if not latest_topic_df.empty else 0

    risk_keywords = ["보안", "해킹", "개인정보", "랜섬웨어", "침해", "취약점"]
    risk_today_count = len(filter_keywords(latest_df, risk_keywords))

    # 오늘의 메가 트렌드는 동시출현 최상위 조합 기준
    if not net_df.empty:
        top_pair = net_df.iloc[0]
        mega_trend = f"{top_pair['keyword_a']} · {top_pair['keyword_b']}"
        mega_desc = f"동시출현 {int(top_pair['co_count']):,}건"
    else:
        mega_trend = "-"
        mega_desc = "동시출현 데이터 없음"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("오늘 최다 키워드", top_kw, f"{top_kw_count:,}건 언급")
    with c2:
        card("오늘 핵심 주제", top_topic, f"{top_topic_count:,}건 관련 기사")
    with c3:
        card("오늘 메가 트렌드", mega_trend, mega_desc)
    with c4:
        card("오늘 리스크 이슈", "보안 / 개인정보", f"{risk_today_count:,}건 탐지", "#ef4444")

    section("오늘의 주요 IT 키워드 트렌드 TOP 10")
    keyword_chip_grid(top_keywords, "keyword", "count", None, clickable=True, session_key="home_drill_keyword")

    if "home_drill_keyword" in st.session_state and st.session_state["home_drill_keyword"]:
        drill_kw = st.session_state["home_drill_keyword"]
        drill_df = filter_keyword(df, drill_kw)
        st.markdown("### 선택 키워드 관련 뉴스")
        st.info(f"선택된 키워드: {drill_kw} / 관련 기사 수: {len(drill_df):,}건")
        article_table(drill_df, DATE_COL)

    section("Today’s Insights")

    ai_semi_count = 0
    risk_count = 0
    cloud_count = 0

    if not latest_df.empty:
        ai_semi_count = len(filter_keywords(latest_df, ["AI반도체", "AI 반도체", "HBM", "GPU", "엔비디아", "반도체"]))
        risk_count = len(filter_keywords(latest_df, ["보안", "해킹", "개인정보", "랜섬웨어", "침해", "취약점"]))
        cloud_count = len(filter_keywords(latest_df, ["클라우드", "AWS", "Azure", "데이터센터", "서버", "인프라"]))

    insight_cards = [
        {
            "title": "AI와 반도체 결합 이슈 증가",
            "value": f"{ai_semi_count:,}건",
            "desc": "HBM · GPU · AI반도체 관련 보도가 함께 증가하는 흐름",
            "color": "#38bdf8"
        },
        {
            "title": "보안 리스크 지속 확대",
            "value": f"{risk_count:,}건",
            "desc": "개인정보 · 해킹 · 침해 · 취약점 중심의 리스크 보도 감지",
            "color": "#ef4444"
        },
        {
            "title": "클라우드 인프라 확장",
            "value": f"{cloud_count:,}건",
            "desc": "AWS · Azure · 데이터센터 · 서버 인프라 관련 보도 흐름",
            "color": "#22c55e"
        }
    ]

    insight_cols = st.columns(3)

    for col, item in zip(insight_cols, insight_cards):
        with col:
            card(
                item["title"],
                item["value"],
                item["desc"],
                item["color"]
            )

elif menu == "Keyword":
    st.subheader("키워드 분석")
    st.caption("키워드를 선택하면 관련 기사, TF-IDF 중요도, 유사 기사까지 한 번에 확인합니다.")

    keyword_options = top_keywords["keyword"].tolist() if not top_keywords.empty else MAIN_KEYWORDS
    selected_kw = st.selectbox("분석할 키워드 선택", keyword_options, key="keyword_page_select")

    selected_articles = filter_keyword(df, selected_kw)

    k1, k2, k3 = st.columns(3)
    with k1:
        card("선택 키워드", selected_kw, "현재 분석 기준 키워드")
    with k2:
        card("관련 기사 수", f"{len(selected_articles):,}건", "전체 기간 기준")
    with k3:
        latest_selected = filter_keyword(latest_df, selected_kw)
        card("오늘 관련 기사", f"{len(latest_selected):,}건", f"{latest_date} 기준")

    section("TF-IDF 기반 주요 IT 키워드", "IT 키워드 후보군 안에서 중요도를 계산합니다.")
    tfidf_df = tfidf_keywords(latest_df, 20)
    st.dataframe(tfidf_df, use_container_width=True)
    if not tfidf_df.empty:
        progress_list(tfidf_df.head(10), "keyword", "article_count", "TF-IDF 기반 주요 IT 키워드 TOP 10")

    section(f"'{selected_kw}' 관련 기사")
    article_table(selected_articles, DATE_COL)

    section(f"'{selected_kw}'와 유사한 기사", "TF-IDF 벡터 간 코사인 유사도 기반입니다.")
    st.dataframe(similar_articles(df, selected_kw), use_container_width=True)

elif menu == "Trend":
    section("일별 주요 IT 키워드", "날짜별 TOP 키워드 변화로 이슈 흐름을 확인합니다.")
    st.dataframe(daily_kw, use_container_width=True)
    sel_kw = st.selectbox("키워드별 일자 추이 확인", MAIN_KEYWORDS)
    st.dataframe(daily_kw[daily_kw["keyword"] == sel_kw], use_container_width=True)

    section("주제별 시계열 트렌드")
    st.dataframe(topic_ts_df, use_container_width=True)
    sel_topic = st.selectbox("시계열 상세 확인 주제", list(TOPIC_MAP.keys()), key="timeseries_topic")
    st.dataframe(topic_ts_df[["date", sel_topic]].sort_values(sel_topic, ascending=False), use_container_width=True)

    section("이벤트 주석 기반 시계열")
    st.dataframe(event_annotations(df, DATE_COL), use_container_width=True)

elif menu == "Network":
    section("키워드 동시출현 네트워크 분석")
    st.dataframe(net_df.head(30), use_container_width=True)

    section("키워드 네트워크 그래프 시각화")
    if not net_df.empty:
        components.html(network_html(net_df), height=760, scrolling=True)
    else:
        st.warning("네트워크 그래프를 생성할 수 있는 데이터가 없습니다.")

elif menu == "Source":
    section("뉴스 수집 소스 기준 분석")
    source_count = df["analysis_source"].value_counts().reset_index()
    source_count.columns = ["source", "count"]
    progress_list(source_count, "source", "count", "수집 소스별 기사 수", top_n=12)

    section("소스별 주요 키워드")
    st.dataframe(source_keyword_table(df), use_container_width=True)
    selected_source = st.selectbox("소스 선택", source_count["source"].tolist(), key="source_select")
    selected_source_df = df[df["analysis_source"] == selected_source]
    progress_list(keyword_counts(selected_source_df, MAIN_KEYWORDS).head(10), "keyword", "count", f"{selected_source} 주요 키워드 TOP 10")

    section("소스별 보도 주제 비중")
    st.dataframe(source_topic_ratio(df), use_container_width=True)

    section("소스별 보도 프레임 분석")
    frame_df = source_frame(df)
    st.dataframe(frame_df, use_container_width=True)
    frame = st.selectbox("프레임 선택", ["성장/혁신", "보안/리스크", "산업/경쟁", "정책/규제", "인프라/클라우드"])
    if frame in frame_df.columns:
        progress_list(frame_df[["source", frame]].sort_values(frame, ascending=False), "source", frame, f"{frame} 프레임 TOP 10")

elif menu == "Sentiment":
    section("주제별 감성 지수 교차 분석")
    st.dataframe(topic_sent_df, use_container_width=True)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(topic_sent_df.sort_values("risk_ratio", ascending=False).head(3).iterrows()):
        with cols[idx]:
            card(row["topic"], f"{row['risk_ratio']}%", "부정/리스크 보도 비율", "#ef4444")

    section("감성/리스크 전체 분포")
    sentiment_count = df["sentiment_group"].value_counts().reset_index()
    sentiment_count.columns = ["sentiment", "count"]
    cols = st.columns(len(sentiment_count))
    color_map = {"긍정/성장": "#22c55e", "중립": "#38bdf8", "부정/리스크": "#ef4444"}
    for idx, (_, row) in enumerate(sentiment_count.iterrows()):
        with cols[idx]:
            card(row["sentiment"], f"{int(row['count']):,}건", "제목/요약문 기반", color_map.get(row["sentiment"], "#38bdf8"))

    with st.expander("부정/리스크 기사 보기"):
        article_table(df[df["sentiment_group"] == "부정/리스크"], DATE_COL)

elif menu == "Search":
    section("전체 기사 검색")
    q = st.text_input("검색어를 입력하세요")
    if q:
        result = filter_keyword(df, q)
        st.write(f"검색 결과: {len(result):,}건")
        article_table(result, DATE_COL)

    with st.expander("원본 데이터 보기"):
        st.dataframe(df, use_container_width=True)
    with st.expander("불러온 파일 목록"):
        for k in sorted(keys):
            st.write(k)
