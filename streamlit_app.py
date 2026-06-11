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

st.set_page_config(
    page_title="IT 뉴스 데이터마이닝 분석 대시보드",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a 0%,#111827 45%,#020617 100%); color:#e5e7eb;}
.block-container {padding-top:1.8rem; padding-bottom:3rem;}
h1 {color:#f8fafc; font-weight:900; letter-spacing:-0.05em;}
h2,h3 {color:#e5e7eb; font-weight:850;}
[data-testid="stCaptionContainer"] {color:#94a3b8;}
[data-testid="stDataFrame"] {border-radius:16px; overflow:hidden; border:1px solid rgba(148,163,184,.18);}
.stAlert {background:rgba(14,165,233,.12); border:1px solid rgba(56,189,248,.35); border-radius:16px; color:#e0f2fe;}
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
st.caption(
    "뉴스 데이터를 단순 나열하지 않고, 분석 질문별로 데이터마이닝 기법을 적용해 "
    "핵심 발견과 해석을 도출합니다."
)

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
    "may", "assumed", "errors", "translation", "translated", "nbsp", "quot", "amp",
    "뉴스", "기사", "기자", "보도", "관련", "대한", "위해", "통해", "이번", "최근", "오늘",
    "있다", "했다", "한다", "밝혔다", "설명했다", "말했다", "지난", "올해", "현재", "가운데",
    "것", "수", "등", "및", "일", "월", "년", "전", "후", "중", "개", "명"
]

MEDIA_DOMAIN_STOPWORDS = [
    "naver", "google", "daum", "v.daum.net", "daum.net", "news.google.com",
    "zdnet", "zdnet.co.kr", "지디넷", "지디넷코리아",
    "ddaily", "ddaily.co.kr", "디지털데일리",
    "digitaltoday", "digitaltoday.co.kr", "디지털투데이",
    "aitimes", "aitimes.com", "AI타임스", "에이아이타임스",
    "boannews", "boannews.com", "보안뉴스",
    "itworld", "itworld.co.kr", "ITWorld", "아이티월드",
    "itnews", "itnews.or.kr", "IT뉴스",
    "ciokorea", "ciokorea.com", "CIO코리아",
    "techworld", "epnc.co.kr", "테크월드",
    "datanet", "datanet.co.kr", "데이터넷",
    "etnews", "etnews.com", "전자신문",
    "www", "com", "co", "kr", "or", "net"
]

TECH_KEYWORD_HINTS = [
    "AI", "인공지능", "생성형", "LLM", "GPT", "OpenAI", "챗GPT", "에이전트",
    "반도체", "HBM", "GPU", "엔비디아", "NVIDIA", "삼성전자", "SK하이닉스", "파운드리", "메모리", "칩",
    "클라우드", "AWS", "Azure", "데이터센터", "서버", "인프라", "SaaS", "쿠버네티스",
    "보안", "해킹", "개인정보", "랜섬웨어", "침해", "취약점", "사이버", "정보보호",
    "로봇", "전기차", "배터리", "자율주행", "모빌리티", "이차전지",
    "플랫폼", "빅테크", "데이터", "빅데이터", "소프트웨어", "디지털", "DX",
    "핀테크", "블록체인", "가상자산", "메타버스", "오픈소스"
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
        label = str(row[label_col])
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


def normalize_token(token):
    token = str(token).strip()
    lower = token.lower()

    alias_map = {
        "ai": "AI",
        "a.i": "AI",
        "gpt": "GPT",
        "chatgpt": "ChatGPT",
        "openai": "OpenAI",
        "llm": "LLM",
        "gpu": "GPU",
        "hbm": "HBM",
        "aws": "AWS",
        "azure": "Azure",
        "nvidia": "NVIDIA",
        "dx": "DX",
        "saas": "SaaS"
    }

    return alias_map.get(lower, token)


def is_noise_token(token):
    token = str(token).strip()
    lower = token.lower()

    if not token:
        return True

    if lower in {s.lower() for s in STOPWORDS}:
        return True

    if lower in {s.lower() for s in MEDIA_DOMAIN_STOPWORDS}:
        return True

    if re.search(r"(https?|www|\.com|\.co\.kr|\.net|\.or\.kr|\.kr)", lower):
        return True

    if re.fullmatch(r"\d+", token):
        return True

    if re.fullmatch(r"\d{1,4}[./-]\d{1,2}([./-]\d{1,2})?", token):
        return True

    if re.fullmatch(r"[가-힣]{1}", token):
        return True

    return False


def clean_for_vectorize(text):
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&quot;", " ").replace("&amp;", " ")
    text = re.sub(r"\b[a-zA-Z0-9.-]+\.(com|co\.kr|net|or\.kr|kr)\b", " ", text)
    text = re.sub(r"[^가-힣A-Za-z0-9\s+#.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for raw in text.split():
        token = normalize_token(raw)
        if not is_noise_token(token):
            tokens.append(token)

    return " ".join(tokens)


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


def safe_top(df, col, default="-"):
    if df is None or df.empty or col not in df.columns:
        return default
    return df.iloc[0][col]

def format_change_rate(previous_count, current_count):
    previous_count = int(previous_count)
    current_count = int(current_count)
    diff = current_count - previous_count

    if previous_count == 0:
        if current_count > 0:
            return "신규 등장"
        return "변화 없음"

    rate = round(diff / previous_count * 100, 1)
    sign = "+" if rate > 0 else ""
    return f"{sign}{rate}%"

# =========================================================
# Analysis Functions
# =========================================================

def tfidf_keywords(df, top_n=25):
    """
    TF-IDF 특징 기술 키워드 추출.
    title + description만 사용하고, 언론사명/도메인명/URL을 제거한다.
    """
    docs = text_series(df).map(clean_for_vectorize).tolist()
    docs = [d for d in docs if d.strip()]

    if len(docs) < 2:
        return pd.DataFrame(columns=["keyword", "tfidf_score", "doc_count"])

    vectorizer = TfidfVectorizer(
        max_features=2000,
        lowercase=False,
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9+#.-]{2,}\b",
        min_df=2
    )

    try:
        mat = vectorizer.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(columns=["keyword", "tfidf_score", "doc_count"])

    scores = mat.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    doc_counts = (mat > 0).sum(axis=0).A1

    media_stop = {s.lower() for s in MEDIA_DOMAIN_STOPWORDS}
    stop = {s.lower() for s in STOPWORDS}
    tech_hints_lower = [k.lower() for k in TECH_KEYWORD_HINTS]

    rows = []
    for term, score, doc_count in zip(terms, scores, doc_counts):
        keyword = normalize_token(term)
        lower = keyword.lower()

        if lower in stop or lower in media_stop:
            continue
        if is_noise_token(keyword):
            continue

        # 도메인성/매체성 단어 재차 제거
        if any(m in lower for m in media_stop if "." in m):
            continue

        # 일반 영문 소문자 단어는 기술 힌트가 아니면 제거
        if re.fullmatch(r"[a-z]{2,}", keyword) and lower not in tech_hints_lower:
            continue

        is_tech_related = any(h in lower for h in tech_hints_lower)
        is_korean_term = bool(re.search(r"[가-힣]{2,}", keyword))
        is_upper_abbrev = bool(re.fullmatch(r"[A-Z0-9+#.-]{2,}", keyword))

        if not (is_tech_related or is_korean_term or is_upper_abbrev):
            continue

        rows.append({
            "keyword": keyword,
            "tfidf_score": round(float(score), 4),
            "doc_count": int(doc_count)
        })

    if not rows:
        return pd.DataFrame(columns=["keyword", "tfidf_score", "doc_count"])

    result = (
        pd.DataFrame(rows)
        .groupby("keyword", as_index=False)
        .agg({"tfidf_score": "sum", "doc_count": "sum"})
        .sort_values("tfidf_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return result


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


def infer_topic_name(top_words):
    text = str(top_words)
    if contains_any(text, ["AI", "인공지능", "GPT", "OpenAI", "LLM", "엔비디아", "GPU"]):
        return "AI 생태계 추정"
    if contains_any(text, ["반도체", "HBM", "삼성전자", "SK하이닉스", "칩", "메모리"]):
        return "반도체 산업 추정"
    if contains_any(text, ["클라우드", "AWS", "Azure", "데이터센터", "서버", "인프라"]):
        return "클라우드/인프라 추정"
    if contains_any(text, ["보안", "해킹", "개인정보", "랜섬웨어", "취약점", "침해"]):
        return "보안/리스크 추정"
    if contains_any(text, ["네이버", "카카오", "구글", "애플", "메타", "플랫폼"]):
        return "플랫폼/빅테크 추정"
    return "기타 토픽"



def compute_article_similarity(df, top_n=30, threshold=0.45):
    """
    TF-IDF + Cosine Similarity 기반 유사 기사 분석.
    제목+요약문을 벡터화한 뒤 기사 간 코사인 유사도를 계산한다.
    threshold 이상인 기사쌍을 유사 기사 후보로 본다.
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=[
            "article_a_index", "article_b_index", "similarity",
            "date_a", "source_a", "title_a",
            "date_b", "source_b", "title_b"
        ])

    working_df = df.reset_index(drop=True).copy()
    docs = text_series(working_df).map(clean_for_vectorize).tolist()

    valid_pairs = [(idx, doc) for idx, doc in enumerate(docs) if str(doc).strip()]
    if len(valid_pairs) < 2:
        return pd.DataFrame(columns=[
            "article_a_index", "article_b_index", "similarity",
            "date_a", "source_a", "title_a",
            "date_b", "source_b", "title_b"
        ])

    valid_indices = [idx for idx, _ in valid_pairs]
    valid_docs = [doc for _, doc in valid_pairs]

    vectorizer = TfidfVectorizer(
        max_features=1500,
        stop_words=STOPWORDS,
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9+#.-]{2,}\b"
    )

    try:
        mat = vectorizer.fit_transform(valid_docs)
        sim = cosine_similarity(mat)
    except ValueError:
        return pd.DataFrame(columns=[
            "article_a_index", "article_b_index", "similarity",
            "date_a", "source_a", "title_a",
            "date_b", "source_b", "title_b"
        ])

    rows = []
    n = sim.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim[i, j])
            if score >= threshold:
                idx_a = valid_indices[i]
                idx_b = valid_indices[j]
                a = working_df.iloc[idx_a]
                b = working_df.iloc[idx_b]

                rows.append({
                    "article_a_index": idx_a,
                    "article_b_index": idx_b,
                    "similarity": round(score, 4),
                    "date_a": a.get("analysis_date", ""),
                    "source_a": a.get("analysis_source", ""),
                    "title_a": a.get("title", ""),
                    "date_b": b.get("analysis_date", ""),
                    "source_b": b.get("analysis_source", ""),
                    "title_b": b.get("title", "")
                })

    if not rows:
        return pd.DataFrame(columns=[
            "article_a_index", "article_b_index", "similarity",
            "date_a", "source_a", "title_a",
            "date_b", "source_b", "title_b"
        ])

    return (
        pd.DataFrame(rows)
        .sort_values("similarity", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def similarity_summary(sim_df, article_count):
    """
    유사 기사 분석 결과 요약.
    """
    if sim_df.empty or article_count == 0:
        return {
            "similar_pair_count": 0,
            "max_similarity": 0,
            "avg_similarity": 0,
            "involved_articles": 0,
            "involved_ratio": 0
        }

    involved = set(sim_df["article_a_index"].tolist()) | set(sim_df["article_b_index"].tolist())

    return {
        "similar_pair_count": len(sim_df),
        "max_similarity": round(float(sim_df["similarity"].max()), 4),
        "avg_similarity": round(float(sim_df["similarity"].mean()), 4),
        "involved_articles": len(involved),
        "involved_ratio": round(len(involved) / article_count * 100, 1) if article_count else 0
    }


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

# =========================================================
# Sidebar Global Filter
# =========================================================

st.sidebar.header("분석 기준 설정")

period_option = st.sidebar.selectbox(
    "분석 단위 선택",
    ["일별", "주별", "월별", "연별"],
    index=0,
    help="선택한 단위에 따라 IT 뉴스 브리핑과 분석 결과의 기준 기간이 바뀝니다."
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
    f"현재 브리핑 기간: {current_period_label}\n\n"
    f"현재 구간 기사 수: {len(active_df):,}건\n\n"
    f"전체 기사 수: {len(df):,}건"
)

# Current period metrics
active_keyword_count = keyword_counts(active_df, CORE_KEYWORDS)
overall_keyword_count = keyword_counts(df, CORE_KEYWORDS)
active_tfidf = tfidf_keywords(active_df, top_n=20)
active_topics = topic_counts(active_df)
keyword_change_df = period_keyword_change(active_df, previous_df, CORE_KEYWORDS)

# Co-occurrence should use frequency/TF-IDF relevant keywords together
network_keywords = []
if not active_keyword_count.empty:
    network_keywords += active_keyword_count["keyword"].head(20).tolist()
if not active_tfidf.empty:
    network_keywords += active_tfidf["keyword"].head(10).tolist()
network_keywords = list(dict.fromkeys([kw for kw in network_keywords if kw]))
if not network_keywords:
    network_keywords = CORE_KEYWORDS[:30]

net_df = keyword_network(active_df if len(active_df) else df, network_keywords)

active_similarity_df = compute_article_similarity(active_df, top_n=50, threshold=0.45)
active_similarity_summary = similarity_summary(active_similarity_df, len(active_df))

lda_docs = text_series(active_df if len(active_df) >= 20 else df).tolist()
lda_topic_df, lda_doc_topics = lda_topic_modeling(lda_docs, n_topics=5, n_words=8)
if not lda_topic_df.empty:
    lda_topic_df["estimated_topic_name"] = lda_topic_df["top_words"].apply(infer_topic_name)
    lda_topic_df = lda_topic_df.sort_values("article_count", ascending=False).reset_index(drop=True)

# Summary findings
top_frequency_keyword = safe_top(active_keyword_count, "keyword")
top_frequency_count = int(safe_top(active_keyword_count, "count", 0)) if not active_keyword_count.empty else 0

top_tfidf_keyword = safe_top(active_tfidf, "keyword")
top_tfidf_score = safe_top(active_tfidf, "tfidf_score", 0)

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

similar_pair_count = active_similarity_summary["similar_pair_count"]
similar_involved_ratio = active_similarity_summary["involved_ratio"]
max_similarity = active_similarity_summary["max_similarity"]

top_lda_topic = safe_top(lda_topic_df, "estimated_topic_name")
top_lda_words = safe_top(lda_topic_df, "top_words")

# =========================================================
# Tabs
# =========================================================

tab_summary, tab_importance, tab_trend, tab_risk, tab_network, tab_lda, tab_similarity, tab_evidence = st.tabs([
    "1. IT 뉴스 브리핑",
    "2. 많이 언급된 키워드 vs 특징 키워드",
    "3. 급부상 이슈 분석",
    "4. 주제별 리스크 분석",
    "5. 함께 움직이는 기술 관계",
    "6. LDA 잠재 토픽 탐색",
    "7. 유사 기사 분석",
    "8. 근거 기사"
])

# =========================================================
# 1. IT News Briefing
# =========================================================

with tab_summary:
    analysis_header(
        "선택한 기간 동안 IT 뉴스에서 어떤 이슈가 있었는가?",
        "IT News Briefing + Descriptive Analytics + TF-IDF + Time Series + Co-occurrence + LDA + Cosine Similarity",
        "일/주/월/연 기준으로 선택한 기간의 뉴스를 요약하여, 해당 기간에 주목할 만한 IT 이슈를 한눈에 확인합니다.",
        "기사 수, 최다 언급 키워드, 특징 키워드, 급부상 키워드, 리스크 신호, 기술 관계, 잠재 토픽, 유사 기사 반복 여부"
    )

    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("브리핑 기간", current_period_label, f"{period_option} 기준 / 원본 {min_date} ~ {max_date}")
    with c2:
        card("브리핑 기사 수", f"{len(active_df):,}건", f"전체 수집 기사 {len(df):,}건")
    with c3:
        card("최다 언급 키워드", top_frequency_keyword, f"{top_frequency_count:,}건")
    with c4:
        card("리스크 기사 비중", f"{risk_ratio}%", f"리스크 기사 {len(risk_df):,}건", "#ef4444")

    s1, s2, s3 = st.columns(3)
    with s1:
        card("유사 기사쌍", f"{similar_pair_count:,}쌍", "TF-IDF + Cosine Similarity 기준")
    with s2:
        card("유사 기사 포함 비율", f"{similar_involved_ratio}%", "현재 구간 기사 중 유사 기사 후보 비율")
    with s3:
        card("최대 유사도", f"{max_similarity}", "가장 유사한 기사쌍의 cosine similarity")

    section("IT 뉴스 브리핑 핵심 요약")
    insight_box("브리핑 요약", [
        f"브리핑 기간 동안 최다 언급 이슈는 '{top_frequency_keyword}'입니다. 현재 구간에서 {top_frequency_count:,}건 등장했습니다.",
        f"TF-IDF 기준으로 이 기간을 특징짓는 키워드는 '{top_tfidf_keyword}'입니다. 단순 빈도와 별도로 현재 뉴스 구간을 특징짓는 단어입니다.",
        f"직전 기간 대비 가장 크게 증가한 키워드는 '{top_change_keyword}'이며 변화량은 {top_change_value:+,}건입니다.",
        f"주제 분류 기준으로 가장 큰 비중을 차지한 분야는 '{top_topic}'이며 전체의 {top_topic_ratio}%입니다.",
        f"기술 관계 분석 기준 가장 강한 연결 관계는 '{top_pair}'이며 {top_pair_count:,}건 함께 등장했습니다.",
        f"LDA 토픽 탐색 결과 주요 잠재 주제는 '{top_lda_topic}'으로 추정되며 대표 단어는 '{top_lda_words}'입니다.",
        f"유사 기사 분석 결과 브리핑 기간 내 유사 기사쌍은 {similar_pair_count:,}쌍이며, 유사 기사 후보에 포함된 기사 비율은 {similar_involved_ratio}%입니다."
    ])

    section("브리핑 근거 데이터")
    col_a, col_b = st.columns(2)
    with col_a:
        progress_list(active_keyword_count.head(10), "keyword", "count", "빈도 기준 핵심 키워드")
    with col_b:
        progress_list(active_topics.head(10), "topic", "count", "주제 분류 결과")

# =========================================================
# 2. Keyword Importance
# =========================================================

with tab_importance:
    analysis_header(
        "많이 나온 키워드와 현재 뉴스를 특징짓는 키워드는 다른가?",
        "Keyword Frequency Analysis + TF-IDF Analysis",
        "단순 빈도는 사전에 정한 IT 키워드가 얼마나 많이 등장했는지 보여주고, TF-IDF는 제목+요약문 전체에서 출처·도메인명을 제거한 뒤 현재 구간을 특징짓는 기술 키워드를 자동 추출합니다.",
        "빈도 TOP 키워드와 TF-IDF TOP 단어를 비교해 많이 언급된 이슈와 특징적인 이슈를 구분합니다."
    )

    freq_top = active_keyword_count.head(10).copy()
    tfidf_top = active_tfidf.head(10).copy()

    f1, f2, f3 = st.columns(3)
    with f1:
        card("빈도 1위", top_frequency_keyword, f"{top_frequency_count:,}건 등장")
    with f2:
        card("TF-IDF 1위", top_tfidf_keyword, f"점수 {top_tfidf_score}")
    with f3:
        same_or_diff = "같음" if top_frequency_keyword == top_tfidf_keyword else "다름"
        card("두 결과 비교", same_or_diff, "다르면 특징 키워드가 별도로 존재한다는 의미")

    section("단순 빈도 TOP 10 vs TF-IDF 특징 기술 키워드 TOP 10")
    col1, col2 = st.columns(2)
    with col1:
        progress_list(freq_top, "keyword", "count", "단순 빈도 TOP 10")
        st.dataframe(freq_top, use_container_width=True)
    with col2:
        progress_list(tfidf_top, "keyword", "tfidf_score", "TF-IDF 특징 기술 키워드 TOP 10", suffix="점")
        st.dataframe(tfidf_top, use_container_width=True)

    freq_set = set(freq_top["keyword"].astype(str).tolist()) if not freq_top.empty else set()
    tfidf_set = set(tfidf_top["keyword"].astype(str).tolist()) if not tfidf_top.empty else set()
    common = sorted(freq_set.intersection(tfidf_set))
    only_tfidf = [kw for kw in tfidf_top["keyword"].astype(str).tolist() if kw not in freq_set] if not tfidf_top.empty else []

    insight_box("분석 해석", [
        f"빈도 1위는 '{top_frequency_keyword}', TF-IDF 1위는 '{top_tfidf_keyword}'입니다.",
        f"빈도 TOP 10과 TF-IDF TOP 10의 공통 키워드는 {len(common)}개입니다.",
        f"TF-IDF에만 강하게 나타난 특징 기술 키워드는 {', '.join(only_tfidf[:5]) if only_tfidf else '없음'}입니다.",
        "이 결과를 통해 단순히 많이 언급된 키워드와 현재 기사 묶음을 특징짓는 단어를 구분할 수 있습니다."
    ])

    if not tfidf_top.empty:
        selected_kw = st.selectbox("TF-IDF 특징 기술 키워드 근거 기사 확인", tfidf_top["keyword"].tolist())
        selected_df = filter_keyword(active_df, selected_kw)
        st.info(f"'{selected_kw}' 관련 기사: {len(selected_df):,}건")
        article_table(selected_df, DATE_COL, n=100)

# =========================================================
# 3. Trend
# =========================================================

with tab_trend:
    analysis_header(
        "직전 기간 대비 어떤 이슈가 급부상하거나 감소했는가?",
        "Time Series Analysis + Period-over-Period Change Analysis",
        "현재 기간과 직전 기간의 키워드 빈도를 비교하여 증가한 이슈, 새롭게 등장한 이슈, 감소한 이슈를 함께 분석합니다.",
        "증가 키워드, 감소 키워드, 신규 등장 키워드, 변화량 및 변화율"
    )

    period_count_df = period_counts(summary_df, PERIOD_COL, period_label_map)
    peak_period = period_count_df.sort_values("article_count", ascending=False).iloc[0]["period_label"] if not period_count_df.empty else "-"
    peak_count = int(period_count_df.sort_values("article_count", ascending=False).iloc[0]["article_count"]) if not period_count_df.empty else 0

    increase_df = keyword_change_df[keyword_change_df["change"] > 0].copy()
    decrease_df = keyword_change_df[keyword_change_df["change"] < 0].copy().sort_values("change")
    new_keyword_df = keyword_change_df[(keyword_change_df["previous_count"] == 0) & (keyword_change_df["current_count"] > 0)].copy()

    if not keyword_change_df.empty:
        keyword_change_df["change_rate_text"] = keyword_change_df.apply(
            lambda r: format_change_rate(r["previous_count"], r["current_count"]),
            axis=1
        )
    if not increase_df.empty:
        increase_df["change_rate_text"] = increase_df.apply(
            lambda r: format_change_rate(r["previous_count"], r["current_count"]),
            axis=1
        )
    if not decrease_df.empty:
        decrease_df["change_rate_text"] = decrease_df.apply(
            lambda r: format_change_rate(r["previous_count"], r["current_count"]),
            axis=1
        )
    if not new_keyword_df.empty:
        new_keyword_df["change_rate_text"] = "신규 등장"

    top_increase = increase_df.iloc[0] if not increase_df.empty else None
    top_decrease = decrease_df.iloc[0] if not decrease_df.empty else None
    top_new_keywords = new_keyword_df["keyword"].head(3).tolist() if not new_keyword_df.empty else []

    top_increase_keyword = top_increase["keyword"] if top_increase is not None else "-"
    top_increase_prev = int(top_increase["previous_count"]) if top_increase is not None else 0
    top_increase_curr = int(top_increase["current_count"]) if top_increase is not None else 0
    top_increase_diff = int(top_increase["change"]) if top_increase is not None else 0
    top_increase_rate = format_change_rate(top_increase_prev, top_increase_curr) if top_increase is not None else "-"

    top_decrease_keyword = top_decrease["keyword"] if top_decrease is not None else "-"
    top_decrease_prev = int(top_decrease["previous_count"]) if top_decrease is not None else 0
    top_decrease_curr = int(top_decrease["current_count"]) if top_decrease is not None else 0
    top_decrease_diff = int(top_decrease["change"]) if top_decrease is not None else 0
    top_decrease_rate = format_change_rate(top_decrease_prev, top_decrease_curr) if top_decrease is not None else "-"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("현재 기간", current_period_label, period_option)
    with c2:
        card("최대 증가 키워드", top_increase_keyword, f"{top_increase_prev:,}건 → {top_increase_curr:,}건 / {top_increase_diff:+,}건")
    with c3:
        card("최대 감소 키워드", top_decrease_keyword, f"{top_decrease_prev:,}건 → {top_decrease_curr:,}건 / {top_decrease_diff:+,}건", "#ef4444")
    with c4:
        card("신규 등장 키워드", f"{len(new_keyword_df):,}개", ", ".join(top_new_keywords) if top_new_keywords else "신규 등장 없음")

    section("증가/감소 키워드")
    col1, col2 = st.columns(2)
    with col1:
        progress_list(increase_df.head(10), "keyword", "change", "증가 키워드 TOP 10", suffix="건")
        if not increase_df.empty:
            st.dataframe(
                increase_df[["keyword", "previous_count", "current_count", "change", "change_rate_text"]].head(10),
                use_container_width=True
            )
        else:
            st.warning("증가한 키워드가 없습니다.")

    with col2:
        decrease_display = decrease_df.copy()
        if not decrease_display.empty:
            decrease_display["decrease_abs"] = decrease_display["change"].abs()
            progress_list(decrease_display.head(10), "keyword", "decrease_abs", "감소 키워드 TOP 10", suffix="건")
            st.dataframe(
                decrease_display[["keyword", "previous_count", "current_count", "change", "change_rate_text"]].head(10),
                use_container_width=True
            )
        else:
            st.warning("감소한 키워드가 없습니다.")

    section("신규 등장 키워드")
    if new_keyword_df.empty:
        st.info("직전 기간에는 없었고 현재 기간에 새롭게 등장한 키워드는 없습니다.")
    else:
        st.dataframe(
            new_keyword_df[["keyword", "previous_count", "current_count", "change", "change_rate_text"]].head(20),
            use_container_width=True
        )

    section("전체 변화량 상세")
    if keyword_change_df.empty:
        st.warning("변화량을 계산할 키워드 데이터가 없습니다.")
    else:
        st.dataframe(
            keyword_change_df[["keyword", "previous_count", "current_count", "change", "change_rate_text"]].sort_values("change", ascending=False),
            use_container_width=True
        )

    insight_items = [
        f"현재 분석 단위는 '{period_option}'이며 현재 기간은 '{current_period_label}'입니다."
    ]

    if top_increase is not None:
        insight_items.append(
            f"'{top_increase_keyword}' 키워드는 직전 기간 대비 가장 큰 증가를 보였습니다. "
            f"({top_increase_prev:,}건 → {top_increase_curr:,}건, {top_increase_diff:+,}건 / {top_increase_rate})"
        )
    else:
        insight_items.append("직전 기간 대비 증가한 키워드는 확인되지 않았습니다.")

    if top_new_keywords:
        insight_items.append(
            f"새롭게 등장한 주요 키워드는 {', '.join(top_new_keywords)}입니다."
        )
    else:
        insight_items.append("새롭게 등장한 키워드는 확인되지 않았습니다.")

    if top_decrease is not None:
        insight_items.append(
            f"'{top_decrease_keyword}' 키워드는 가장 큰 감소를 보였습니다. "
            f"({top_decrease_prev:,}건 → {top_decrease_curr:,}건, {top_decrease_diff:+,}건 / {top_decrease_rate})"
        )
    else:
        insight_items.append("직전 기간 대비 감소한 키워드는 확인되지 않았습니다.")

    insight_items.append(
        "증가 키워드는 단기적으로 관심이 상승한 이슈이며, 감소 키워드는 직전 기간 대비 관심이 줄어든 이슈로 해석할 수 있습니다."
    )

    insight_box("분석 해석", insight_items)

# =========================================================
# 4. Risk
# =========================================================

with tab_risk:
    analysis_header(
        "어떤 IT 주제에서 리스크 신호가 강한가?",
        "Rule-based Sentiment Analysis + Risk Keyword Analysis",
        "긍정/성장 키워드와 부정/리스크 키워드를 기준으로 기사 성향을 분류하고, 주제별 리스크 비율을 계산합니다.",
        "전체 감성 분포, 주제별 부정/리스크 비율, 리스크 기사 근거"
    )

    sentiment_count = active_df["sentiment_group"].value_counts().reset_index()
    sentiment_count.columns = ["sentiment", "count"] if not sentiment_count.empty else ["sentiment", "count"]
    if not sentiment_count.empty:
        sentiment_count["ratio"] = (sentiment_count["count"] / len(active_df) * 100).round(1)

    neg_count = int(sentiment_count[sentiment_count["sentiment"] == "부정/리스크"]["count"].sum()) if not sentiment_count.empty else 0
    pos_count = int(sentiment_count[sentiment_count["sentiment"] == "긍정/성장"]["count"].sum()) if not sentiment_count.empty else 0
    neg_ratio = round(neg_count / len(active_df) * 100, 1) if len(active_df) else 0

    if len(active_df):
        topic_risk = active_df.pivot_table(
            index="primary_topic",
            columns="sentiment_group",
            values="title",
            aggfunc="count",
            fill_value=0
        ).reset_index()
        sentiment_cols = [c for c in ["긍정/성장", "중립", "부정/리스크"] if c in topic_risk.columns]
        topic_risk["total"] = topic_risk[sentiment_cols].sum(axis=1) if sentiment_cols else 0
        if "부정/리스크" not in topic_risk.columns:
            topic_risk["부정/리스크"] = 0
        topic_risk["risk_ratio"] = (topic_risk["부정/리스크"] / topic_risk["total"] * 100).round(1).fillna(0)
        topic_risk = topic_risk.sort_values("risk_ratio", ascending=False)
    else:
        topic_risk = pd.DataFrame(columns=["primary_topic", "risk_ratio", "total"])

    top_risk_topic = safe_top(topic_risk, "primary_topic")
    top_risk_ratio = safe_top(topic_risk, "risk_ratio", 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        card("대표 성향", safe_top(sentiment_count, "sentiment"), "현재 구간 기준")
    with c2:
        card("부정/리스크 기사", f"{neg_count:,}건", f"전체 대비 {neg_ratio}%", "#ef4444")
    with c3:
        card("리스크 최상위 주제", top_risk_topic, f"리스크 비율 {top_risk_ratio}%")

    section("감성 분포 및 주제별 리스크")
    col1, col2 = st.columns(2)
    with col1:
        progress_list(sentiment_count, "sentiment", "count", "현재 구간 감성 분포")
    with col2:
        progress_list(topic_risk, "primary_topic", "risk_ratio", "주제별 리스크 비율", suffix="%")

    st.dataframe(topic_risk, use_container_width=True)

    insight_box("분석 해석", [
        f"현재 구간의 부정/리스크 기사는 {neg_count:,}건이며 전체 대비 {neg_ratio}%입니다.",
        f"리스크 비율이 가장 높은 주제는 '{top_risk_topic}'이며 리스크 비율은 {top_risk_ratio}%입니다.",
        "이 분석은 단순히 부정 기사가 몇 건인지가 아니라, 어떤 IT 분야에서 위험 신호가 강한지를 확인하기 위한 것입니다."
    ])

    with st.expander("리스크 기사 근거 보기"):
        article_table(active_df[active_df["sentiment_group"] == "부정/리스크"], DATE_COL, n=150)

# =========================================================
# 5. Co-occurrence
# =========================================================

with tab_network:
    analysis_header(
        "어떤 기술 이슈들이 함께 움직이는가?",
        "Co-occurrence Analysis + Network Analysis",
        "같은 기사 안에 함께 등장한 키워드 쌍을 계산해 기술 이슈 간 연결 구조를 확인합니다.",
        "가장 강한 연결 관계, 네트워크 구조, 동시출현 빈도"
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
            "예를 들어 AI와 반도체가 함께 등장한다면, AI 이슈가 소프트웨어를 넘어 GPU, HBM, 반도체 산업과 함께 보도되고 있음을 의미할 수 있습니다."
        ])

# =========================================================
# 6. LDA
# =========================================================

with tab_lda:
    analysis_header(
        "뉴스 속 숨겨진 잠재 토픽은 무엇인가?",
        "LDA Topic Modeling",
        "사전에 정한 분류 기준이 아니라, 뉴스 데이터 내부에서 함께 등장하는 단어 패턴을 기반으로 잠재 토픽을 탐색합니다.",
        "토픽별 대표 단어, 추정 토픽명, 토픽별 기사 비중"
    )

    st.info(
        "LDA는 데이터 수와 기간이 충분할수록 안정적인 결과를 냅니다. "
        "현재 결과는 확정적 결론이 아니라 잠재 토픽 탐색 결과로 해석하는 것이 적절합니다."
    )

    lda_topic_count = st.slider("LDA 토픽 수", min_value=3, max_value=8, value=5)
    lda_word_count = st.slider("토픽별 대표 단어 수", min_value=5, max_value=12, value=8)

    lda_topic_df_view, lda_doc_topics_view = lda_topic_modeling(
        lda_docs,
        n_topics=lda_topic_count,
        n_words=lda_word_count,
        max_features=1200
    )

    if lda_topic_df_view.empty:
        st.warning("LDA 분석을 수행하기에 문서 수 또는 단어 수가 부족합니다.")
    else:
        lda_topic_df_view["estimated_topic_name"] = lda_topic_df_view["top_words"].apply(infer_topic_name)
        lda_topic_df_view = lda_topic_df_view.sort_values("article_count", ascending=False).reset_index(drop=True)
        top_lda = lda_topic_df_view.iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            card("최대 잠재 토픽", top_lda["estimated_topic_name"], top_lda["topic_id"])
        with c2:
            card("대표 단어", top_lda["top_words"].split(",")[0], top_lda["top_words"])
        with c3:
            card("토픽 비중", f"{top_lda['ratio']}%", f"{int(top_lda['article_count']):,}건")

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.dataframe(lda_topic_df_view, use_container_width=True)
        with col2:
            progress_list(lda_topic_df_view, "estimated_topic_name", "article_count", "LDA 토픽별 기사 수", top_n=lda_topic_count)

        insight_box("분석 해석", [
            f"가장 큰 잠재 토픽은 '{top_lda['estimated_topic_name']}'으로 추정됩니다.",
            f"대표 단어는 '{top_lda['top_words']}'입니다.",
            "LDA 토픽명은 모델이 자동으로 붙이는 것이 아니라 대표 단어를 보고 사람이 해석한 추정명입니다.",
            "현재 수집 기간이 짧기 때문에 향후 데이터가 누적되면 토픽 안정성을 다시 검증해야 합니다."
        ])


# =========================================================
# 7. Similarity
# =========================================================

with tab_similarity:
    analysis_header(
        "제목은 다르지만 내용이 유사한 반복 기사는 얼마나 있는가?",
        "TF-IDF Vectorization + Cosine Similarity",
        "뉴스 데이터는 같은 이슈가 여러 매체에서 반복 보도되는 경우가 많다. 단순 기사 수만 보면 실제 정보 다양성이 과대평가될 수 있으므로 기사 간 유사도를 계산한다.",
        "유사 기사쌍, 최대 유사도, 유사 기사 포함 비율, 유사 기사 근거"
    )

    threshold = st.slider(
        "유사도 기준값",
        min_value=0.20,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="값이 낮을수록 더 많은 기사쌍이 유사 기사 후보로 잡히고, 값이 높을수록 매우 비슷한 기사만 잡힙니다."
    )

    sim_df = compute_article_similarity(active_df, top_n=100, threshold=threshold)
    sim_summary = similarity_summary(sim_df, len(active_df))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("유사 기사쌍", f"{sim_summary['similar_pair_count']:,}쌍", f"유사도 {threshold} 이상")
    with c2:
        card("유사 기사 포함 비율", f"{sim_summary['involved_ratio']}%", "현재 구간 기사 중 유사 기사 후보 비율")
    with c3:
        card("최대 유사도", f"{sim_summary['max_similarity']}", "가장 유사한 기사쌍")
    with c4:
        card("평균 유사도", f"{sim_summary['avg_similarity']}", "탐지된 유사 기사쌍 평균")

    section("유사 기사 분석 결과")
    if sim_df.empty:
        st.warning("현재 기준값 이상으로 유사한 기사쌍이 탐지되지 않았습니다. 기준값을 낮추면 더 많은 후보를 확인할 수 있습니다.")
    else:
        st.dataframe(sim_df, use_container_width=True)

        top_sim = sim_df.iloc[0]
        insight_box("분석 해석", [
            f"현재 분석 구간에서 유사도 {threshold} 이상인 기사쌍은 {sim_summary['similar_pair_count']:,}쌍입니다.",
            f"가장 높은 유사도는 {sim_summary['max_similarity']}이며, 이는 제목이나 요약문이 매우 비슷한 기사쌍이 존재한다는 의미입니다.",
            f"유사 기사 후보에 포함된 기사는 전체의 {sim_summary['involved_ratio']}%입니다.",
            "이 분석은 단순 기사 수가 실제 정보 다양성을 그대로 의미하지 않을 수 있음을 확인하기 위한 보완 분석입니다.",
            "따라서 뉴스 데이터 분석 시 중복 기사뿐 아니라 내용이 유사한 반복 기사도 고려해야 합니다."
        ])

        section("가장 유사한 기사쌍 예시")
        st.markdown("### 기사 A")
        st.write(f"출처: {top_sim['source_a']} / 날짜: {top_sim['date_a']}")
        st.write(top_sim["title_a"])

        st.markdown("### 기사 B")
        st.write(f"출처: {top_sim['source_b']} / 날짜: {top_sim['date_b']}")
        st.write(top_sim["title_b"])

    section("분석 의미")
    insight_box("왜 이 분석을 추가했는가?", [
        "기존에는 동일 제목 또는 동일 링크 중심으로 중복을 제거했지만, 제목이 조금 다르고 내용이 비슷한 기사는 남을 수 있습니다.",
        "TF-IDF는 기사 제목과 요약문을 벡터로 변환하고, Cosine Similarity는 두 기사 벡터의 방향이 얼마나 비슷한지 계산합니다.",
        "이를 통해 유사 기사 반복 문제를 한계점으로만 남기지 않고, 대시보드에서 직접 탐지할 수 있도록 보완했습니다."
    ])



# =========================================================
# 8. Evidence
# =========================================================

with tab_evidence:
    analysis_header(
        "분석 결과의 근거 기사는 무엇인가?",
        "Evidence-based Validation",
        "분석 결과가 단순 수치에 그치지 않도록 실제 기사 제목, 요약, 링크를 확인합니다.",
        "키워드, 주제, 감성, 소스별 근거 기사"
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
