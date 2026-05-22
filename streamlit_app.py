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
</style>
""", unsafe_allow_html=True)

st.title("IT News Intelligence Dashboard")
st.caption("IT 뉴스 텍스트마이닝 기반 키워드 트렌드 · 소스 분석 · 유사도 · 네트워크 · 감성/리스크 분석")

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

STOPWORDS = [
    "the", "and", "for", "that", "with", "this", "from", "have", "will", "into", "about",
    "their", "they", "them", "were", "been", "being", "said", "more", "than", "over",
    "after", "before", "while", "where", "when", "what", "which", "would", "could", "should",
    "there", "these", "those", "because", "through", "during", "under", "between", "among",
    "it", "to", "of", "in", "is", "on", "at", "by", "be", "as", "an", "or", "if", "we",
    "he", "she", "you", "are", "was", "has", "had", "can", "not", "also", "but", "how",
    "news", "media", "report", "reports", "update", "service", "platform", "technology", "tech",
    "company", "companies", "market", "business", "press", "article",
    "daum", "net", "com", "kr", "www", "naver", "google", "rss", "site",
    "trump", "president", "government", "white", "house",
    "관련", "통해", "기자", "이번", "대한", "위한", "있는", "있다", "한다", "했다", "하는", "등의", "등을",
    "에서", "으로", "까지", "지난", "오늘", "올해", "최근", "발표", "제공", "추진", "확대",
    "서비스", "기업", "기술", "산업", "시장", "뉴스", "정보기술",
    "ai", "it"
]

ALLOW_ENGLISH_TERMS = {
    "aws", "azure", "hbm", "gpu", "nvidia", "openai", "chatgpt", "gpt", "llm", "skt", "kt",
    "lg", "sk", "ms", "meta", "apple", "cloud", "security", "data", "server", "semiconductor"
}

# =========================================================
# UI helpers
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
        padding:18px 22px;margin-bottom:14px;box-shadow:0 0 18px rgba(56,189,248,.05);">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <div style="font-size:20px;font-weight:900;color:white;">#{idx+1} {label}</div>
            <div style="font-size:19px;font-weight:900;color:{color};">{value_text}</div>
          </div>
          <div style="width:100%;height:14px;background:#1e293b;border-radius:999px;overflow:hidden;">
            <div style="width:{ratio}%;height:100%;background:linear-gradient(90deg,{color},#38bdf8);border-radius:999px;box-shadow:0 0 14px {color};"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


def methodology_cards():
    methods = [
        ("TF-IDF", "흔한 단어보다 특정 기사군에서 상대적으로 중요한 단어에 높은 가중치를 부여합니다."),
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
# Data utilities
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

    for col in ["source", "source_group", "media_domain", "title", "description", "originallink", "link"]:
        if col not in df.columns:
            df[col] = ""

    df["source"] = df["source"].fillna("").astype(str)
    df.loc[df["source"].str.strip() == "", "source"] = "unknown"

    df["source_group"] = df["source_group"].fillna("").astype(str)
    empty_group_mask = df["source_group"].str.strip() == ""
    df.loc[empty_group_mask, "source_group"] = df.loc[empty_group_mask, "source"]

    # 현재 데이터 품질상 media_domain보다 source 기준 분석이 더 신뢰 가능함
    df["analysis_source"] = df["source"].fillna("").astype(str)
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
# Analysis functions
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


def is_valid_tfidf_keyword(keyword):
    keyword = str(keyword).strip().lower()

    if not keyword or keyword in STOPWORDS:
        return False

    parts = keyword.split()

    if any(part in STOPWORDS for part in parts):
        return False

    # 영어만 있는 일반 단어는 제거하고, IT 용어 화이트리스트는 유지
    if re.fullmatch("[a-z0-9 ]+", keyword):
        compact = keyword.replace(" ", "")
        if compact not in ALLOW_ENGLISH_TERMS:
            return False

    if re.fullmatch("[0-9]+", keyword):
        return False

    if len(keyword.replace(" ", "")) < 2:
        return False

    return True


def tfidf_keywords(df, top_n=20):
    """
    자유 단어 추출형 TF-IDF가 아니라 IT 키워드 사전 기반 TF-IDF.
    뉴스 문장에는 the, daum, net, 있다, 넘어 같은 노이즈가 많기 때문에
    분석 목적에 맞는 IT 후보 키워드 안에서만 중요도를 계산한다.
    """
    txt = text_series(df).fillna("").astype(str)

    if len(txt) == 0 or txt.str.strip().eq("").all():
        return pd.DataFrame(columns=["keyword", "score", "article_count"])

    candidate_keywords = [
        # AI
        "AI", "인공지능", "생성형AI", "생성형 AI", "LLM", "챗GPT", "ChatGPT", "GPT", "OpenAI",
        "멀티모달", "AI 에이전트", "AI 검색", "AI 서비스",

        # Semiconductor
        "반도체", "AI반도체", "AI 반도체", "HBM", "GPU", "엔비디아", "NVIDIA",
        "삼성전자", "SK하이닉스", "파운드리", "메모리", "칩", "첨단 반도체",

        # Cloud / Infra
        "클라우드", "AWS", "Azure", "데이터센터", "데이터 센터", "서버", "인프라",
        "SaaS", "쿠버네티스", "Kubernetes", "클라우드 전환",

        # Security
        "보안", "해킹", "개인정보", "랜섬웨어", "침해", "취약점", "사이버 공격",
        "정보보호", "제로트러스트", "인증", "망분리",

        # Mobility / Robot / Battery
        "로봇", "전기차", "배터리", "자율주행", "모빌리티", "이차전지", "전장",

        # Platform / Big Tech
        "네이버", "카카오", "구글", "애플", "메타", "마이크로소프트", "MS",
        "플랫폼", "빅테크", "검색", "커머스",

        # Data / Software
        "데이터", "빅데이터", "소프트웨어", "디지털", "DX", "디지털전환",
        "핀테크", "블록체인", "가상자산", "메타버스"
    ]

    total_docs = len(txt)
    rows = []

    for keyword in candidate_keywords:
        contains = txt.str.contains(keyword, case=False, regex=False)
        article_count = int(contains.sum())

        if article_count == 0:
            continue

        # 후보 키워드가 등장한 기사 수 기반 TF-IDF 유사 점수
        # 많이 등장하되, 모든 기사에 흔하게 퍼진 키워드는 과도하게 커지지 않도록 보정
        tf = article_count
        idf = 1 + pd.np.log((1 + total_docs) / (1 + article_count)) if hasattr(pd, "np") else 1 + __import__("math").log((1 + total_docs) / (1 + article_count))
        score = tf * idf

        rows.append({
            "keyword": keyword,
            "score": round(score, 4),
            "article_count": article_count
        })

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
        sub, txt = df[df["analysis_source"] == src], text_series(df[df["analysis_source"] == src])
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
# Load + compute
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
# Dashboard
# =========================================================

st.subheader("오늘의 IT 뉴스 요약")
c1, c2, c3, c4 = st.columns(4)
with c1: card("전체 기사 수", f"{len(df):,}", "중복 제거 후 분석 대상 기사")
with c2: card(f"{latest_date} 기사 수", f"{len(latest_df):,}", "최신일 기준 수집 기사")
with c3: card("수집 소스 수", f"{df['analysis_source'].nunique():,}", "source 기준")
with c4: card("수집 파일 수", f"{len(keys):,}", "S3 processed CSV 파일")

section("분석 방법론", "뉴스 텍스트마이닝 기반의 5가지 분석 방법입니다.")
methodology_cards()
st.info("본 대시보드는 source 기준 분석을 사용합니다. 현재 데이터 구조상 media_domain 누락이 많아 실제 언론사보다 수집 소스 기준이 더 신뢰 가능합니다.")

section("1. 오늘의 주요 IT 키워드 트렌드")
progress_list(top_keywords, "keyword", "count", "오늘의 주요 IT 키워드 TOP 10")

section("2. 뉴스 수집 소스 기준 분석", "네이버 API, Google News, RSS, Google site 검색 등 수집 경로별 보도 경향을 비교합니다.")
source_count = df["analysis_source"].value_counts().reset_index()
source_count.columns = ["source", "count"]
progress_list(source_count, "source", "count", "수집 소스별 기사 수", top_n=12)
st.dataframe(source_keyword_table(df), use_container_width=True)

if not source_count.empty:
    selected_source = st.selectbox("소스 선택", source_count["source"].tolist(), key="source_select")
    selected_source_df = df[df["analysis_source"] == selected_source]
    progress_list(keyword_counts(selected_source_df, MAIN_KEYWORDS).head(10), "keyword", "count", f"{selected_source} 주요 키워드 TOP 10")
    with st.expander(f"{selected_source} 기사 보기"):
        article_table(selected_source_df, DATE_COL)

section("3. TF-IDF 기반 핵심 키워드 분석", "불용어를 제거하고 특정 기사군에서 상대적으로 중요한 키워드를 추출합니다.")
tfidf_df = tfidf_keywords(latest_df, 20)
st.dataframe(tfidf_df, use_container_width=True)
if not tfidf_df.empty:
    tfidf_view = tfidf_df.assign(score_view=(tfidf_df["score"] * 100).round(2))
    progress_list(tfidf_view.head(10), "keyword", "score_view", "TF-IDF 중요 키워드 TOP 10", suffix="")

section("4. Cosine Similarity 기반 유사 기사 분석", "TF-IDF 벡터 간 각도 유사도로 선택 키워드와 가까운 기사를 찾습니다.")
sim_kw = st.selectbox("유사 기사 분석 키워드 선택", MAIN_KEYWORDS, key="similarity_keyword")
st.dataframe(similar_articles(df, sim_kw), use_container_width=True)

section("5. 일별 주요 IT 키워드", "날짜별 TOP 키워드 변화로 이슈 흐름을 확인합니다.")
st.dataframe(daily_kw, use_container_width=True)
sel_kw = st.selectbox("키워드별 일자 추이 확인", MAIN_KEYWORDS)
st.dataframe(daily_kw[daily_kw["keyword"] == sel_kw], use_container_width=True)

section("6. 키워드 클릭해서 관련 기사 보기")
if "selected_keyword" not in st.session_state:
    st.session_state["selected_keyword"] = top_keywords.iloc[0]["keyword"] if not top_keywords.empty else ""
cols = st.columns(5)
for idx, row in top_keywords.reset_index(drop=True).iterrows():
    with cols[idx % 5]:
        if st.button(f"{row['keyword']} ({row['count']})"):
            st.session_state["selected_keyword"] = row["keyword"]
if st.session_state["selected_keyword"]:
    sub = filter_keyword(df, st.session_state["selected_keyword"])
    st.info(f"선택된 키워드: {st.session_state['selected_keyword']} / 관련 기사 수: {len(sub):,}건")
    article_table(sub, DATE_COL)

section(f"7. {latest_date} 주요 이슈 자동 요약", "키워드를 주제군으로 묶어 최신일 이슈 구조를 보여줍니다.")
cols = st.columns(3)
for idx, (_, row) in enumerate(topic_df.iterrows()):
    with cols[idx % 3]:
        card(row["topic"], f"{int(row['count']):,}건", row["keywords"])
st.dataframe(topic_df, use_container_width=True)

section("8. 키워드 동시출현 네트워크 분석", "같은 기사 안에 함께 등장한 키워드 조합을 계산합니다.")
st.dataframe(net_df.head(30), use_container_width=True)

section("9. 키워드 네트워크 그래프 시각화", "동시출현 관계를 네트워크 그래픽으로 표현합니다.")
if not net_df.empty:
    components.html(network_html(net_df), height=760, scrolling=True)
else:
    st.warning("네트워크 그래프를 생성할 수 있는 데이터가 없습니다.")

section("10. 주제별 감성 지수 교차 분석", "어떤 기술 주제가 긍정적/리스크 중심으로 보도되는지 비교합니다.")
st.dataframe(topic_sent_df, use_container_width=True)
cols = st.columns(3)
for idx, (_, row) in enumerate(topic_sent_df.sort_values("risk_ratio", ascending=False).head(3).iterrows()):
    with cols[idx]:
        card(row["topic"], f"{row['risk_ratio']}%", "부정/리스크 보도 비율", "#ef4444")

section("11. 주제별 시계열 트렌드", "날짜별 주제 기사 수 변화를 통해 이슈의 생애주기를 봅니다.")
st.dataframe(topic_ts_df, use_container_width=True)
sel_topic = st.selectbox("시계열 상세 확인 주제", list(TOPIC_MAP.keys()), key="timeseries_topic")
st.dataframe(topic_ts_df[["date", sel_topic]].sort_values(sel_topic, ascending=False), use_container_width=True)

section("12. 소스별 보도 주제 비중", "각 수집 소스가 어떤 IT 주제를 많이 다루는지 비교합니다.")
st.dataframe(source_topic_ratio(df), use_container_width=True)

section("13. 소스별 보도 프레임 분석", "성장/혁신, 보안/리스크, 산업/경쟁 등 보도 관점 차이를 비교합니다.")
frame_df = source_frame(df)
st.dataframe(frame_df, use_container_width=True)
frame = st.selectbox("프레임 선택", ["성장/혁신", "보안/리스크", "산업/경쟁", "정책/규제", "인프라/클라우드"])
if frame in frame_df.columns:
    progress_list(frame_df[["source", frame]].sort_values(frame, ascending=False), "source", frame, f"{frame} 프레임 TOP 10")

section("보조 분석: 감성/리스크 전체 분포")
sentiment_count = df["sentiment_group"].value_counts().reset_index()
sentiment_count.columns = ["sentiment", "count"]
cols = st.columns(len(sentiment_count))
color_map = {"긍정/성장": "#22c55e", "중립": "#38bdf8", "부정/리스크": "#ef4444"}
for idx, (_, row) in enumerate(sentiment_count.iterrows()):
    with cols[idx]:
        card(row["sentiment"], f"{int(row['count']):,}건", "제목/요약문 기반", color_map.get(row["sentiment"], "#38bdf8"))
with st.expander("부정/리스크 기사 보기"):
    article_table(df[df["sentiment_group"] == "부정/리스크"], DATE_COL)

section("보조 분석: 이벤트 주석 기반 시계열")
st.dataframe(event_annotations(df, DATE_COL), use_container_width=True)

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
