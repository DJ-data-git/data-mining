import re
import tempfile
from datetime import datetime, timedelta
from itertools import combinations

import boto3
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 💡 형태소 분석기 추가 (Streamlit Cloud 배포 시 packages.txt에 default-jre 추가 필수)
from konlpy.tag import Okt

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
[data-testid="stSidebar"] {background-color: #0f172a; border-right: 1px solid #334155;}
[data-testid="stCaptionContainer"] {color:#94a3b8;}
[data-testid="stDataFrame"] {border-radius:16px; overflow:hidden; border:1px solid rgba(148,163,184,.18);}
.stAlert {background:rgba(14,165,233,.12); border:1px solid rgba(56,189,248,.35); border-radius:16px; color:#e0f2fe;}
hr {border:none; height:1px; background:linear-gradient(90deg,transparent,#38bdf8,transparent); margin:2rem 0;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Config & Models
# =========================================================

AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = st.secrets["AWS_REGION"]
BUCKET_NAME = st.secrets["BUCKET_NAME"]
S3_PREFIX = "it_news/IT/processed/"

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
    "뉴스", "기자", "오늘", "최근", "올해", "관련", "위한", "대한", "통해", 
    "이번", "서비스", "기업", "기술", "산업", "시장", "발표", "추진", "제공"
]

# 형태소 분석기 로드 최적화 (캐싱)
@st.cache_resource
def get_okt():
    return Okt()

def custom_tokenizer(text):
    okt = get_okt()
    # 명사만 추출하되 2글자 이상만 사용
    return [word for word in okt.nouns(text) if len(word) > 1]

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
    if title: st.markdown(f"### {title}")
    if df.empty: return st.warning("표시할 데이터가 없습니다.")

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
        padding:18px 22px;margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <div style="font-size:20px;font-weight:900;color:white;">#{idx+1} {label}</div>
            <div style="font-size:19px;font-weight:900;color:{color};">{value_text}</div>
          </div>
          <div style="width:100%;height:14px;background:#1e293b;border-radius:999px;overflow:hidden;">
            <div style="width:{ratio}%;height:100%;background:linear-gradient(90deg,{color},#38bdf8);border-radius:999px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# Data utilities (Optimized)
# =========================================================

@st.cache_resource
def get_s3_client():
    return boto3.client(
        "s3", aws_access_key_id=AWS_ACCESS_KEY_ID, 
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION
    )

@st.cache_data(ttl=600)
def fetch_s3_data(bucket, prefix):
    s3 = get_s3_client()
    files, token = [], None
    while True:
        params = {"Bucket": bucket, "Prefix": prefix}
        if token: params["ContinuationToken"] = token
        res = s3.list_objects_v2(**params)
        files.extend(res.get("Contents", []))
        if not res.get("IsTruncated"): break
        token = res.get("NextContinuationToken")
    
    keys = [f["Key"] for f in files if f["Key"].endswith(".csv")]
    dfs = []
    for key in keys:
        obj = s3.get_object(Bucket=bucket, Key=key)
        tmp = pd.read_csv(obj["Body"])
        dfs.append(tmp)
        
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return prepare_df(df)

def prepare_df(df):
    if df.empty: return df, "analysis_date"
    
    date_col = next((c for c in ["pubDate_dt", "pubDate_ymd", "pubDate", "date", "published_date"] if c in df.columns), None)
    df["analysis_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d").fillna("unknown") if date_col else "unknown"

    df["analysis_source"] = df.get("source", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    
    # 💡 텍스트 합치기 최적화 (이후 연산에서 반복 수행 방지)
    df["combined_text"] = df.get("title", "").fillna("").astype(str) + " " + df.get("description", "").fillna("").astype(str)
    
    dedup_cols = [c for c in ["originallink", "link", "title"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last")

    # 💡 감성 분석 카운트 방식 최적화 (존재 유무 -> 실제 빈도수 기반)
    pos_words = ["성장", "확대", "출시", "투자", "협력", "개선", "강화", "수주", "증가", "성공", "최초", "혁신"]
    neg_words = ["해킹", "침해", "유출", "장애", "중단", "규제", "감소", "적자", "위험", "논란", "피해", "공격"]
    
    def get_sentiment(text):
        pos = sum(text.count(w) for w in pos_words)
        neg = sum(text.count(w) for w in neg_words)
        return "긍정/성장" if pos > neg else ("부정/리스크" if neg > pos else "중립")
        
    df["sentiment_group"] = df["combined_text"].apply(get_sentiment)
    
    return df, "analysis_date"

# =========================================================
# Analysis functions (Optimized)
# =========================================================

def keyword_counts(df, keywords):
    # apply 대신 str.contains 누적으로 연산 속도 개선
    txt = df["combined_text"]
    counts = {kw: txt.str.contains(kw, case=False, regex=False).sum() for kw in keywords}
    return pd.DataFrame(list(counts.items()), columns=["keyword", "count"]).sort_values("count", ascending=False)

def tfidf_keywords(text_list, top_n=20):
    if not text_list: return pd.DataFrame()
    # 💡 KoNLPy 커스텀 토크나이저 적용
    vec = TfidfVectorizer(max_features=1500, tokenizer=custom_tokenizer, stop_words=STOPWORDS)
    try:
        mat = vec.fit_transform(text_list)
        return pd.DataFrame({"keyword": vec.get_feature_names_out(), "score": mat.sum(axis=0).A1}).sort_values("score", ascending=False).head(top_n)
    except ValueError:
        return pd.DataFrame()

def similar_articles(df, keyword, top_n=10):
    txt_list = df["combined_text"].tolist()
    if len(txt_list) < 2: return pd.DataFrame()
    
    vec = TfidfVectorizer(max_features=1200, tokenizer=custom_tokenizer, stop_words=STOPWORDS)
    try:
        mat = vec.fit_transform(txt_list + [keyword])
        scores = cosine_similarity(mat[:-1], mat[-1]).flatten()
    except ValueError:
        return pd.DataFrame()
        
    out = df.copy()
    out["similarity_score"] = scores
    cols = [c for c in ["analysis_date", "analysis_source", "title", "link", "similarity_score"] if c in out.columns]
    return out.sort_values("similarity_score", ascending=False)[cols].head(top_n)

def keyword_network(df):
    rows = []
    # 미리 소문자로 변환된 키워드 세트 준비
    lower_kws = {kw: kw.lower() for kw in MAIN_KEYWORDS}
    for text in df["combined_text"]:
        text_lower = text.lower()
        appeared = sorted([kw for kw, lower_kw in lower_kws.items() if lower_kw in text_lower])
        rows.extend(combinations(appeared, 2))
        
    if not rows: return pd.DataFrame(columns=["keyword_a", "keyword_b", "co_count"])
    return pd.DataFrame(rows, columns=["keyword_a", "keyword_b"]).value_counts().reset_index(name="co_count")

# =========================================================
# Main Dashboard
# =========================================================

st.title("IT News Intelligence Dashboard")
st.caption("IT 뉴스 텍스트마이닝 기반 키워드 트렌드 · 소스 분석 · 유사도 · 네트워크 · 감성 분석")

# 전체 데이터 로드
raw_df, DATE_COL = fetch_s3_data(BUCKET_NAME, S3_PREFIX)

if raw_df.empty:
    st.error("S3에서 데이터를 불러오지 못했습니다.")
    st.stop()

# 💡 사이드바 UI 최적화
with st.sidebar:
    st.header("⚙️ 분석 필터")
    
    # 날짜 필터
    min_date = pd.to_datetime(raw_df[DATE_COL].min())
    max_date = pd.to_datetime(raw_df[DATE_COL].max())
    
    date_range = st.date_input("분석 기간 선택", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_dt, end_dt = date_range
    else:
        start_dt = end_dt = date_range[0]

    # 소스 필터
    sources = ["전체"] + raw_df["analysis_source"].unique().tolist()
    selected_source = st.selectbox("뉴스 소스 선택", sources)

# 사이드바 필터 적용
mask = (pd.to_datetime(raw_df[DATE_COL]) >= pd.to_datetime(start_dt)) & (pd.to_datetime(raw_df[DATE_COL]) <= pd.to_datetime(end_dt))
df = raw_df[mask]

if selected_source != "전체":
    df = df[df["analysis_source"] == selected_source]

if df.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

latest_date = df[DATE_COL].max()
latest_df = df[df[DATE_COL] == latest_date]

# =========================================================
# Render UI
# =========================================================

c1, c2, c3 = st.columns(3)
with c1: card("선택 기간 기사 수", f"{len(df):,}")
with c2: card(f"최신일({latest_date}) 기사", f"{len(latest_df):,}")
with c3: card("수집 소스 수", f"{df['analysis_source'].nunique():,}")

section("1. 주요 IT 키워드 트렌드 (선택 기간 기준)")
top_keywords = keyword_counts(df, MAIN_KEYWORDS).head(10)
progress_list(top_keywords, "keyword", "count")

section("2. TF-IDF 기반 핵심 키워드 분석", "형태소 분석기(KoNLPy)를 적용해 명사 단위로 문서 내 중요도를 추출합니다.")
with st.spinner("TF-IDF 연산 중..."):
    tfidf_df = tfidf_keywords(df["combined_text"].tolist(), 15)
    if not tfidf_df.empty:
        tfidf_df["score"] = (tfidf_df["score"] * 100).round(2)
        progress_list(tfidf_df, "keyword", "score", suffix="점")

section("3. Cosine Similarity 기반 유사 기사 분석")
sim_kw = st.selectbox("분석할 키워드를 선택하세요", MAIN_KEYWORDS)
st.dataframe(similar_articles(df, sim_kw), use_container_width=True)

section("4. 키워드 동시출현 네트워크 분석")
net_df = keyword_network(df)
st.dataframe(net_df.head(10), use_container_width=True)
if not net_df.empty:
    # 네트워크 생성 로직 (pyvis) - 간소화 적용
    G = nx.Graph()
    for _, row in net_df.head(25).iterrows():
        G.add_edge(row["keyword_a"], row["keyword_b"], value=int(row["co_count"]))
    net = Network(height="600px", width="100%", bgcolor="#0f172a", font_color="white")
    net.from_nx(G)
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    net.save_graph(path)
    with open(path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=620)

section("5. 감성/리스크 분석 요약 (개선된 빈도 기반)")
sentiment_count = df["sentiment_group"].value_counts().reset_index()
sentiment_count.columns = ["sentiment", "count"]
cols = st.columns(len(sentiment_count))
color_map = {"긍정/성장": "#22c55e", "중립": "#38bdf8", "부정/리스크": "#ef4444"}

for idx, (_, row) in enumerate(sentiment_count.iterrows()):
    with cols[idx]:
        card(row["sentiment"], f"{int(row['count']):,}건", color=color_map.get(row["sentiment"], "#38bdf8"))

section("전체 기사 검색")
q = st.text_input("검색어를 입력하세요 (예: 엔비디아)")
if q:
    result = df[df["combined_text"].str.contains(q, case=False, regex=False)]
    st.write(f"검색 결과: {len(result):,}건")
    st.dataframe(result[["analysis_date", "analysis_source", "title", "link"]], use_container_width=True)
