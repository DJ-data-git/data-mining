import re
import tempfile
from datetime import datetime
from itertools import combinations

import boto3
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 💡 형태소 분석기 (Streamlit Cloud 배포 시 packages.txt에 default-jre 추가 필수)
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
.stButton > button {background:linear-gradient(135deg,#0284c7,#2563eb); color:white; border:0; border-radius:999px; padding:.5rem 1rem; font-weight:700;}
.stButton > button:hover {background:linear-gradient(135deg,#38bdf8,#2563eb); color:white; border:0;}
.stTextInput input {background:#020617; color:#e5e7eb; border:1px solid #334155; border-radius:12px;}
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
    "이번", "서비스", "기업", "기술", "산업", "시장", "발표", "추진", "제공", "확대"
]

@st.cache_resource
def get_okt():
    return Okt()

def custom_tokenizer(text):
    okt = get_okt()
    return [word for word in okt.nouns(text) if len(word) > 1]

# =========================================================
# UI helpers
# =========================================================

def section(title, subtitle=None):
    st.markdown("---")
    st.subheader(title)
    if subtitle: st.caption(subtitle)

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
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return st.warning("표시할 데이터가 없습니다.")

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
        tmp["loaded_file"] = key
        dfs.append(tmp)
        
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return prepare_df(df), len(keys), keys

def prepare_df(df):
    if df.empty: return df, "analysis_date"
    
    date_col = next((c for c in ["pubDate_dt", "pubDate_ymd", "pubDate", "date", "published_date"] if c in df.columns), None)
    df["analysis_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d").fillna("unknown") if date_col else "unknown"

    df["analysis_source"] = df.get("source", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    df.loc[df["analysis_source"].str.strip() == "", "analysis_source"] = "unknown"
    
    df["source_group"] = df.get("source_group", df["analysis_source"]).fillna("").astype(str)
    df.loc[df["source_group"].str.strip() == "", "source_group"] = df["analysis_source"]
    
    # 💡 텍스트 합치기 최적화 (이후 연산에서 text_series() 대신 사용)
    df["combined_text"] = df.get("title", "").fillna("").astype(str) + " " + df.get("description", "").fillna("").astype(str)
    
    dedup_cols = [c for c in ["originallink", "link", "title"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last")

    # 💡 감성 분석 카운트 방식 최적화 (존재 유무 -> 실제 빈도수 기반)
    pos_words = ["성장", "확대", "출시", "투자", "협력", "개선", "강화", "수주", "증가", "성공", "최초", "고도화", "혁신"]
    neg_words = ["해킹", "침해", "유출", "장애", "중단", "규제", "감소", "적자", "위험", "논란", "피해", "취약점", "공격"]
    
    def get_sentiment(text):
        pos = sum(text.count(w) for w in pos_words)
        neg = sum(text.count(w) for w in neg_words)
        return "긍정/성장" if pos > neg else ("부정/리스크" if neg > pos else "중립")
        
    df["sentiment_group"] = df["combined_text"].apply(get_sentiment)
    
    return df, "analysis_date"

# =========================================================
# Analysis functions (Using combined_text)
# =========================================================

def filter_keyword(df, keyword):
    return df[df["combined_text"].str.contains(keyword, case=False, regex=False)]

def filter_keywords(df, keywords):
    cond = pd.Series(False, index=df.index)
    for kw in keywords:
        cond |= df["combined_text"].str.contains(kw, case=False, regex=False)
    return df[cond]

def keyword_counts(df, keywords):
    counts = {kw: df["combined_text"].str.contains(kw, case=False, regex=False).sum() for kw in keywords}
    return pd.DataFrame(list(counts.items()), columns=["keyword", "count"]).sort_values("count", ascending=False)

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
    lower_kws = {kw: kw.lower() for kw in MAIN_KEYWORDS}
    for text in df["combined_text"]:
        text_lower = text.lower()
        appeared = sorted([kw for kw, lower_kw in lower_kws.items() if lower_kw in text_lower])
        rows.extend(combinations(appeared, 2))
    if not rows: return pd.DataFrame(columns=["keyword_a", "keyword_b", "co_count"])
    return pd.DataFrame(rows, columns=["keyword_a", "keyword_b"]).value_counts().reset_index(name="co_count")

def tfidf_keywords(text_list, top_n=20):
    if not text_list: return pd.DataFrame()
    vec = TfidfVectorizer(max_features=1500, tokenizer=custom_tokenizer, stop_words=STOPWORDS)
    try:
        mat = vec.fit_transform(text_list)
        return pd.DataFrame({"keyword": vec.get_feature_names_out(), "score": mat.sum(axis=0).A1}).sort_values("score", ascending=False).head(top_n)
    except ValueError:
        return pd.DataFrame(columns=["keyword", "score"])

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
    cols = [c for c in ["analysis_date", "analysis_source", "source_group", "title", "description", "link", "similarity_score"] if c in out.columns]
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
        txt = sub["combined_text"]
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
    cols = [c for c in [date_col, "analysis_source", "source_group", "title", "description", "link"] if c in df.columns]
    if date_col in df.columns:
        df = df.sort_values(date_col, ascending=False)
    st.dataframe(df[cols], use_container_width=True)


# =========================================================
# Main Dashboard Loading
# =========================================================

raw_df, num_files, keys = fetch_s3_data(BUCKET_NAME, S3_PREFIX)

if raw_df.empty:
    st.error("S3에서 데이터를 불러오지 못했습니다.")
    st.stop()
DATE_COL = "analysis_date"

# 💡 사이드바 UI 추가 (선택 사항이었으나 UX 개선을 위해 유지)
with st.sidebar:
    st.header("⚙️ 분석 필터")
    min_date = pd.to_datetime(raw_df[DATE_COL].min())
    max_date = pd.to_datetime(raw_df[DATE_COL].max())
    date_range = st.date_input("분석 기간", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_dt, end_dt = date_range
    else:
        start_dt = end_dt = date_range[0]

    sources = ["전체"] + raw_df["analysis_source"].unique().tolist()
    selected_source_sidebar = st.selectbox("뉴스 소스 필터", sources)

# 사이드바 필터 적용
mask = (pd.to_datetime(raw_df[DATE_COL]) >= pd.to_datetime(start_dt)) & (pd.to_datetime(raw_df[DATE_COL]) <= pd.to_datetime(end_dt))
df = raw_df[mask]

if selected_source_sidebar != "전체":
    df = df[df["analysis_source"] == selected_source_sidebar]

if df.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

latest_date = df[DATE_COL].max()
latest_df = df[df[DATE_COL] == latest_date]

# 연산 수행 (최적화된 함수들 활용)
top_keywords = keyword_counts(latest_df, MAIN_KEYWORDS).head(10)
daily_kw = daily_top_keywords(df, DATE_COL)
net_df = keyword_network(df)
topic_df = topic_counts(latest_df)
topic_sent_df = topic_sentiment(df)
topic_ts_df = topic_timeseries(df, DATE_COL)


# =========================================================
# Dashboard Render (13개 섹션 원상 복구)
# =========================================================

st.title("IT News Intelligence Dashboard")
st.caption("IT 뉴스 텍스트마이닝 기반 키워드 트렌드 · 소스 분석 · 유사도 · 네트워크 · 감성/리스크 분석")

st.subheader(f"{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} 뉴스 요약")
c1, c2, c3, c4 = st.columns(4)
with c1: card("선택 기간 기사 수", f"{len(df):,}", "필터링된 분석 대상 기사")
with c2: card(f"{latest_date} 기사 수", f"{len(latest_df):,}", "선택 기간 내 최신일 기사")
with c3: card("수집 소스 수", f"{df['analysis_source'].nunique():,}", "source 기준")
with c4: card("전체 파일 수", f"{num_files:,}", "S3 전체 CSV 파일")

section("분석 방법론", "뉴스 텍스트마이닝 기반의 5가지 분석 방법입니다.")
methodology_cards()
st.info("본 대시보드는 source 기준 분석을 사용합니다. 현재 데이터 구조상 media_domain 누락이 많아 실제 언론사보다 수집 소스 기준이 더 신뢰 가능합니다.")

section("1. 오늘의 주요 IT 키워드 트렌드")
progress_list(top_keywords, "keyword", "count", "주요 IT 키워드 TOP 10")

section("2. 뉴스 수집 소스 기준 분석", "수집 경로별 보도 경향을 비교합니다.")
source_count = df["analysis_source"].value_counts().reset_index()
source_count.columns = ["source", "count"]
progress_list(source_count, "source", "count", "수집 소스별 기사 수", top_n=12)
st.dataframe(source_keyword_table(df), use_container_width=True)

if not source_count.empty:
    selected_source = st.selectbox("소스 선택 상세 분석", source_count["source"].tolist(), key="source_select_main")
    selected_source_df = df[df["analysis_source"] == selected_source]
    progress_list(keyword_counts(selected_source_df, MAIN_KEYWORDS).head(10), "keyword", "count", f"{selected_source} 주요 키워드 TOP 10")
    with st.expander(f"{selected_source} 기사 보기"):
        article_table(selected_source_df, DATE_COL)

section("3. TF-IDF 기반 핵심 키워드 분석", "불용어를 제거하고 형태소 분석기를 통해 중요한 단어를 추출합니다.")
with st.spinner("TF-IDF 연산 중..."):
    tfidf_df = tfidf_keywords(latest_df["combined_text"].tolist(), 20)
    st.dataframe(tfidf_df, use_container_width=True)
    if not tfidf_df.empty:
        tfidf_view = tfidf_df.assign(score_view=(tfidf_df["score"] * 100).round(2))
        progress_list(tfidf_view.head(10), "keyword", "score_view", "TF-IDF 중요 키워드 TOP 10", suffix="")

section("4. Cosine Similarity 기반 유사 기사 분석", "TF-IDF 벡터 간 각도 유사도로 선택 키워드와 가까운 기사를 찾습니다.")
sim_kw = st.selectbox("유사 기사 분석 키워드 선택", MAIN_KEYWORDS, key="similarity_keyword")
with st.spinner("유사도 계산 중..."):
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
if not topic_ts_df.empty:
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
