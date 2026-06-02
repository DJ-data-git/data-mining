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

st.set_page_config(page_title="IT 뉴스 트렌드 분석 대시보드", layout="wide")

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
/* Premium top tab navigation */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(15,23,42,.62);
    border: 1px solid rgba(148,163,184,.16);
    padding: 10px;
    border-radius: 20px;
    box-shadow: 0 0 24px rgba(56,189,248,.06);
}
.stTabs [data-baseweb="tab"] {
    height: 46px;
    border-radius: 14px;
    padding: 0 18px;
    background: rgba(2,6,23,.45);
    border: 1px solid rgba(148,163,184,.10);
    color: #94a3b8;
    font-weight: 800;
    letter-spacing: -0.01em;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(30,41,59,.8);
    color: #e5e7eb;
    border: 1px solid rgba(56,189,248,.35);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(14,165,233,.95), rgba(37,99,235,.92)) !important;
    color: #ffffff !important;
    border: 1px solid rgba(125,211,252,.8) !important;
    box-shadow: 0 0 20px rgba(56,189,248,.25);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

st.title("IT 뉴스 트렌드 분석 대시보드")

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
    "AI", "인공지능", "생성형AI", "챗GPT", "반도체", "클라우드", "보안", "데이터",
    "로봇", "배터리", "전기차", "삼성", "네이버", "카카오", "엔비디아",
    "AWS", "Azure", "해킹", "개인정보", "데이터센터", "HBM"
]

# MAIN_KEYWORDS는 이후 데이터 기반 자동 키워드와 결합해 하이브리드 방식으로 재정의됨
MAIN_KEYWORDS = CORE_KEYWORDS.copy()

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
    rows = []
    for kw in keywords:
        count = int(txt.str.contains(kw, case=False, regex=False).sum())
        if count > 0:
            rows.append({"keyword": kw, "count": count})
    if not rows:
        return pd.DataFrame(columns=["keyword", "count"])
    return pd.DataFrame(rows).sort_values("count", ascending=False)

# =========================================================
# Analysis Functions
# =========================================================

def extract_dynamic_keywords(df, top_n=20):
    """
    데이터에서 자동으로 떠오르는 키워드를 추출한다.
    단, 무의미한 일반어를 줄이기 위해 IT 후보 키워드 사전 안에서만 추출한다.
    """
    tfidf_df = tfidf_keywords(df, top_n=top_n)
    if tfidf_df.empty:
        return []
    return tfidf_df["keyword"].dropna().astype(str).tolist()


def build_hybrid_keywords(df, core_keywords=None, dynamic_top_n=20, max_total=40):
    """
    Core Keywords + Dynamic Keywords 결합.
    - Core: 분석의 안정성을 위한 고정 IT 핵심 키워드
    - Dynamic: 최신 뉴스 데이터에서 자동 추출된 emerging keyword
    """
    if core_keywords is None:
        core_keywords = CORE_KEYWORDS

    dynamic_keywords = extract_dynamic_keywords(df, top_n=dynamic_top_n)

    merged = []
    for kw in list(core_keywords) + dynamic_keywords:
        kw = str(kw).strip()
        if kw and kw not in merged:
            merged.append(kw)

    return merged[:max_total], dynamic_keywords


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


def render_daily_keyword_timeline(daily_df, top_n=5):
    if daily_df.empty:
        st.warning("일별 키워드 데이터가 없습니다.")
        return

    st.markdown("### 일별 키워드 히트맵")
    st.caption("날짜별로 어떤 IT 키워드가 강하게 등장했는지 한눈에 확인합니다.")

    pivot = daily_df.pivot_table(
        index="keyword",
        columns="date",
        values="count",
        aggfunc="sum",
        fill_value=0
    )

    keyword_order = pivot.sum(axis=1).sort_values(ascending=False).head(12).index
    pivot = pivot.loc[keyword_order]

    st.dataframe(
        pivot.style.background_gradient(axis=1),
        use_container_width=True
    )

    st.markdown("### 날짜별 TOP 키워드 상세")
    selected_date = st.selectbox(
        "상세 확인 날짜 선택",
        sorted(daily_df["date"].dropna().unique(), reverse=True),
        key="daily_keyword_date_select"
    )

    day_df = daily_df[daily_df["date"] == selected_date].sort_values("rank").head(top_n)

    cols = st.columns(top_n)
    prev_date_candidates = [d for d in sorted(daily_df["date"].dropna().unique()) if d < selected_date]
    prev_rank_map = {}
    if prev_date_candidates:
        prev_date = prev_date_candidates[-1]
        prev_df = daily_df[daily_df["date"] == prev_date]
        prev_rank_map = dict(zip(prev_df["keyword"], prev_df["rank"]))

    for idx, (_, row) in enumerate(day_df.iterrows()):
        keyword = row["keyword"]
        count = int(row["count"])
        rank = int(row["rank"])
        prev_rank = prev_rank_map.get(keyword)

        if prev_rank is None:
            movement = "NEW"
            movement_color = "#38bdf8"
        elif prev_rank > rank:
            movement = "UP"
            movement_color = "#22c55e"
        elif prev_rank < rank:
            movement = "DOWN"
            movement_color = "#ef4444"
        else:
            movement = "STAY"
            movement_color = "#94a3b8"

        with cols[idx]:
            st.markdown(f"""
            <div style="
                background:rgba(15,23,42,.82);
                border:1px solid rgba(56,189,248,.20);
                border-radius:18px;
                padding:16px 14px;
                min-height:132px;
                box-shadow:0 0 16px rgba(56,189,248,.05);
            ">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <div style="font-size:13px;color:#94a3b8;font-weight:800;">Rank {rank}</div>
                    <div style="font-size:11px;color:{movement_color};font-weight:950;">{movement}</div>
                </div>
                <div style="font-size:20px;font-weight:950;color:#f8fafc;margin-bottom:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                    {keyword}
                </div>
                <div style="font-size:24px;font-weight:950;color:#38bdf8;margin-bottom:8px;">
                    {count:,}건
                </div>
                <div style="font-size:12px;color:#64748b;line-height:1.4;">
                    선택일 기준 TOP 키워드
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_keyword_trend_detail(daily_df, keyword):
    selected = daily_df[daily_df["keyword"] == keyword].sort_values("date")

    if selected.empty:
        st.warning(f"'{keyword}' 키워드의 일별 데이터가 없습니다.")
        return

    st.markdown(f"### '{keyword}' 일별 움직임")

    cols = st.columns(3)
    with cols[0]:
        card("등장 일수", f"{selected['date'].nunique():,}일", "TOP 키워드에 포함된 날짜 수")
    with cols[1]:
        card("최고 순위", f"{int(selected['rank'].min())}위", "기간 내 가장 높은 순위")
    with cols[2]:
        card("최대 기사 수", f"{int(selected['count'].max()):,}건", "하루 기준 최대 언급량")

    progress_list(
        selected.sort_values("count", ascending=False),
        "date",
        "count",
        f"'{keyword}' 언급량이 높았던 날짜",
        top_n=min(10, len(selected))
    )

    st.dataframe(selected, use_container_width=True)


def topic_timeseries(df, date_col):
    rows = []
    for date in sorted(df[date_col].dropna().unique()):
        sub = df[df[date_col] == date]
        row = {"date": date}
        for topic, kws in TOPIC_MAP.items():
            row[topic] = len(filter_keywords(sub, kws))
        rows.append(row)
    return pd.DataFrame(rows)


def keyword_daily_matrix(df, date_col):
    rows = []

    for date in sorted(df[date_col].dropna().unique()):
        sub = df[df[date_col] == date]
        counts = keyword_counts(sub, MAIN_KEYWORDS)

        for _, row in counts.iterrows():
            rows.append({
                "date": date,
                "keyword": row["keyword"],
                "count": int(row["count"])
            })

    return pd.DataFrame(rows)


def surge_sustain_analysis(df, date_col):
    """
    키워드별 총 언급량, 등장 일수, 최대 증가율을 계산해
    지속형/급등형/이벤트형/관찰형으로 분류한다.
    """
    daily = keyword_daily_matrix(df, date_col)

    if daily.empty:
        return pd.DataFrame(columns=[
            "keyword", "total_count", "active_days", "max_count",
            "avg_count", "max_growth_rate", "trend_type"
        ])

    pivot = daily.pivot_table(
        index="date",
        columns="keyword",
        values="count",
        aggfunc="sum",
        fill_value=0
    ).sort_index()

    rows = []

    for keyword in pivot.columns:
        series = pivot[keyword].astype(float)
        total_count = int(series.sum())
        active_days = int((series > 0).sum())
        max_count = int(series.max())
        avg_count = round(float(series.mean()), 2)

        prev = series.shift(1).replace(0, pd.NA)
        growth = ((series - prev) / prev * 100).replace([float("inf"), -float("inf")], pd.NA)
        max_growth_rate = round(float(growth.max()), 1) if growth.notna().any() else 0.0

        if active_days >= max(3, int(len(series) * 0.6)):
            trend_type = "지속형"
        elif max_growth_rate >= 100:
            trend_type = "급등형"
        elif active_days <= 2 and max_count >= 5:
            trend_type = "이벤트형"
        else:
            trend_type = "관찰형"

        rows.append({
            "keyword": keyword,
            "total_count": total_count,
            "active_days": active_days,
            "max_count": max_count,
            "avg_count": avg_count,
            "max_growth_rate": max_growth_rate,
            "trend_type": trend_type
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(["trend_type", "total_count"], ascending=[True, False]).reset_index(drop=True)


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
        weight = int(row["co_count"])
        G.add_edge(
            row["keyword_a"],
            row["keyword_b"],
            value=max(1, weight / 80),
            width=max(1, min(10, weight / 120)),
            title=f"{row['keyword_a']} ↔ {row['keyword_b']} | 동시출현 {weight:,}건"
        )

    net = Network(
        height="720px",
        width="100%",
        bgcolor="#0f172a",
        font_color="white",
        notebook=False
    )
    net.from_nx(G)

    degrees = dict(G.degree())
    weighted_degree = dict(G.degree(weight="value"))

    for node in net.nodes:
        node_id = node["id"]
        size = 20 + weighted_degree.get(node_id, 1) * 1.8
        node["size"] = min(76, max(24, size))
        node["color"] = {
            "background": "#38bdf8",
            "border": "#7dd3fc",
            "highlight": {"background": "#60a5fa", "border": "#e0f2fe"}
        }
        node["borderWidth"] = 2
        node["title"] = f"{node_id}<br>연결 키워드 수: {degrees.get(node_id, 0)}<br>연결 강도: {weighted_degree.get(node_id, 0):.1f}"
        node["font"] = {"size": 18, "face": "Arial", "color": "#f8fafc", "strokeWidth": 3, "strokeColor": "#0f172a"}

    for edge in net.edges:
        edge["color"] = {"color": "rgba(148,163,184,.72)", "highlight": "#38bdf8"}
        edge["smooth"] = {"type": "dynamic"}

    net.set_options("""
    var options = {
      "interaction": {
        "hover": true,
        "tooltipDelay": 120,
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      },
      "nodes": {
        "shape": "dot",
        "scaling": {"min": 20, "max": 80}
      },
      "edges": {
        "scaling": {"min": 1, "max": 10},
        "smooth": {"enabled": true, "type": "continuous"}
      },
      "physics": {
        "enabled": false
      }
    }
    """)

    path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    net.save_graph(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_strong_connections(net_df, top_n=6):
    if net_df.empty:
        st.warning("표시할 키워드 연결 데이터가 없습니다.")
        return

    st.markdown("### 주요 키워드 연결")
    st.caption("같은 기사 안에서 가장 자주 함께 등장한 키워드 조합입니다.")

    view = net_df.head(top_n).reset_index(drop=True)
    cols = st.columns(3)

    for idx, (_, row) in enumerate(view.iterrows()):
        with cols[idx % 3]:
            pair = f"{row['keyword_a']} ↔ {row['keyword_b']}"
            card(
                pair,
                f"{int(row['co_count']):,}건",
                "동시출현 빈도 기준 핵심 연결",
                "#38bdf8"
            )


def render_network_summary(net_df):
    if net_df.empty:
        return

    top_pair = net_df.iloc[0]
    unique_nodes = len(set(net_df["keyword_a"]).union(set(net_df["keyword_b"])))
    total_links = len(net_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        card("가장 강한 연결", f"{top_pair['keyword_a']} ↔ {top_pair['keyword_b']}", f"{int(top_pair['co_count']):,}건 동시출현")
    with c2:
        card("연결 키워드 수", f"{unique_nodes:,}", "네트워크에 포함된 고유 키워드")
    with c3:
        card("키워드 연결 수", f"{total_links:,}", "분석된 키워드 페어 수")


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

MAIN_KEYWORDS, DYNAMIC_KEYWORDS = build_hybrid_keywords(latest_df, CORE_KEYWORDS, dynamic_top_n=20, max_total=40)

top_keywords = keyword_counts(latest_df, MAIN_KEYWORDS).head(10)
daily_kw = daily_top_keywords(df, DATE_COL)
net_df = keyword_network(df)
topic_df = topic_counts(latest_df)
topic_sent_df = topic_sentiment(df)
topic_ts_df = topic_timeseries(df, DATE_COL)
surge_df = surge_sustain_analysis(df, DATE_COL)


# =========================================================
# Executive Analysis Dashboard
# =========================================================

# 이 버전은 단순 기사 나열보다 "분석 결과 → 근거 데이터 → 관련 기사" 흐름을 우선합니다.

# 추가 분석 유틸리티

def safe_first_value(df, col, default="-"):
    if df is None or df.empty or col not in df.columns:
        return default
    value = df.iloc[0][col]
    return default if pd.isna(value) else value


def daily_article_summary(df, date_col):
    if df.empty or date_col not in df.columns:
        return pd.DataFrame(columns=["date", "article_count", "prev_count", "change", "change_rate"])

    daily = (
        df.groupby(date_col)
        .size()
        .reset_index(name="article_count")
        .sort_values(date_col)
    )
    daily = daily.rename(columns={date_col: "date"})
    daily["prev_count"] = daily["article_count"].shift(1)
    daily["change"] = daily["article_count"] - daily["prev_count"]
    daily["change_rate"] = (daily["change"] / daily["prev_count"].replace(0, pd.NA) * 100).round(1)
    daily[["prev_count", "change", "change_rate"]] = daily[["prev_count", "change", "change_rate"]].fillna(0)
    return daily


def keyword_daily_detail(df, date_col, keyword):
    if not keyword:
        return pd.DataFrame(columns=["date", "count", "prev_count", "change", "change_rate"])

    rows = []
    for date in sorted(df[date_col].dropna().unique()):
        sub = df[df[date_col] == date]
        count = int(text_series(sub).str.contains(keyword, case=False, regex=False).sum())
        rows.append({"date": date, "count": count})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["date", "count", "prev_count", "change", "change_rate"])
    out["prev_count"] = out["count"].shift(1)
    out["change"] = out["count"] - out["prev_count"]
    out["change_rate"] = (out["change"] / out["prev_count"].replace(0, pd.NA) * 100).round(1)
    out[["prev_count", "change", "change_rate"]] = out[["prev_count", "change", "change_rate"]].fillna(0)
    return out


def keyword_profile(df, date_col, keyword):
    daily = keyword_daily_detail(df, date_col, keyword)
    if daily.empty:
        return {
            "total": 0,
            "active_days": 0,
            "peak_date": "-",
            "peak_count": 0,
            "latest_count": 0,
            "latest_change": 0,
            "latest_rate": 0,
            "trend_type": "관찰형",
        }

    total = int(daily["count"].sum())
    active_days = int((daily["count"] > 0).sum())
    peak_row = daily.sort_values("count", ascending=False).iloc[0]
    latest_row = daily.iloc[-1]

    trend_type = "관찰형"
    if not surge_df.empty and keyword in surge_df["keyword"].astype(str).tolist():
        trend_type = surge_df[surge_df["keyword"].astype(str) == str(keyword)].iloc[0]["trend_type"]
    elif active_days >= max(3, int(df[date_col].nunique() * 0.6)):
        trend_type = "지속형"
    elif daily["change_rate"].max() >= 100:
        trend_type = "급등형"

    return {
        "total": total,
        "active_days": active_days,
        "peak_date": peak_row["date"],
        "peak_count": int(peak_row["count"]),
        "latest_count": int(latest_row["count"]),
        "latest_change": int(latest_row["change"]),
        "latest_rate": float(latest_row["change_rate"]),
        "trend_type": trend_type,
    }


def top_keyword_movers(df, date_col, keywords, top_n=10):
    dates = sorted(df[date_col].dropna().unique())
    if len(dates) < 2:
        return pd.DataFrame(columns=["keyword", "latest_count", "prev_count", "change", "change_rate"])

    prev_date, latest_date_local = dates[-2], dates[-1]
    prev_df = df[df[date_col] == prev_date]
    latest_df_local = df[df[date_col] == latest_date_local]

    rows = []
    for kw in keywords:
        prev_count = int(text_series(prev_df).str.contains(kw, case=False, regex=False).sum())
        latest_count = int(text_series(latest_df_local).str.contains(kw, case=False, regex=False).sum())
        change = latest_count - prev_count
        rate = round((change / prev_count * 100), 1) if prev_count else (100.0 if latest_count else 0.0)
        if latest_count > 0 or prev_count > 0:
            rows.append({
                "keyword": kw,
                "latest_count": latest_count,
                "prev_count": prev_count,
                "change": change,
                "change_rate": rate,
            })

    if not rows:
        return pd.DataFrame(columns=["keyword", "latest_count", "prev_count", "change", "change_rate"])
    return pd.DataFrame(rows).sort_values(["change", "latest_count"], ascending=False).head(top_n)


def dominant_topic_by_date(df, date_col):
    rows = []
    for date in sorted(df[date_col].dropna().unique()):
        sub = df[df[date_col] == date]
        topic_count = topic_counts(sub)
        if topic_count.empty:
            continue
        top = topic_count.iloc[0]
        rows.append({
            "date": date,
            "dominant_topic": top["topic"],
            "count": int(top["count"]),
            "keywords": top["keywords"],
        })
    return pd.DataFrame(rows)


def render_insight_box(title, bullets, tone="info"):
    text = "\n".join([f"- {b}" for b in bullets if b])
    if tone == "warning":
        st.warning(f"**{title}**\n\n{text}")
    elif tone == "error":
        st.error(f"**{title}**\n\n{text}")
    else:
        st.info(f"**{title}**\n\n{text}")


def compact_article_evidence(df, keyword=None, top_n=8):
    view = df.copy()
    if keyword:
        view = filter_keyword(view, keyword)
    if view.empty:
        st.warning("관련 기사 근거가 없습니다.")
        return
    cols = [c for c in [DATE_COL, "analysis_source", "title", "description", "originallink"] if c in view.columns]
    st.dataframe(view.sort_values(DATE_COL, ascending=False)[cols].head(top_n), use_container_width=True)


def add_period_column(df, date_col, period_option):
    """
    선택한 분석 단위에 따라 집계용 컬럼과 표시용 컬럼을 생성한다.

    - 일별: 2026-06-02
    - 주별: 2026-06-01 ~ 2026-06-07
    - 월별: 2026년 6월
    - 연별: 2026년
    """
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce")

    if period_option == "일별":
        out["analysis_period"] = dt.dt.strftime("%Y-%m-%d")
        out["analysis_period_label"] = out["analysis_period"]

    elif period_option == "주별":
        week_start = dt - pd.to_timedelta(dt.dt.weekday, unit="D")
        week_end = week_start + pd.Timedelta(days=6)
        out["analysis_period"] = week_start.dt.strftime("%Y-%m-%d")
        out["analysis_period_label"] = (
            week_start.dt.strftime("%Y-%m-%d")
            + " ~ "
            + week_end.dt.strftime("%Y-%m-%d")
        )

    elif period_option == "월별":
        out["analysis_period"] = dt.dt.strftime("%Y-%m")
        out["analysis_period_label"] = (
            dt.dt.year.astype("Int64").astype(str)
            + "년 "
            + dt.dt.month.astype("Int64").astype(str)
            + "월"
        )

    elif period_option == "연별":
        out["analysis_period"] = dt.dt.strftime("%Y")
        out["analysis_period_label"] = dt.dt.year.astype("Int64").astype(str) + "년"

    else:
        out["analysis_period"] = out[date_col].astype(str)
        out["analysis_period_label"] = out["analysis_period"]

    out["analysis_period"] = out["analysis_period"].fillna("unknown").astype(str)
    out["analysis_period_label"] = out["analysis_period_label"].fillna(out["analysis_period"]).astype(str)

    out.loc[out["analysis_period_label"].str.contains("<NA>", na=False), "analysis_period_label"] = "unknown"
    out.loc[out["analysis_period"].isin(["NaT", "nan", "None"]), "analysis_period"] = "unknown"
    out.loc[out["analysis_period_label"].isin(["NaT", "nan", "None"]), "analysis_period_label"] = out["analysis_period"]

    return out


def get_current_period_info(df, date_col, period_option):
    """
    데이터의 최신 날짜를 기준으로 현재 분석 구간의 group value와 label을 반환한다.
    """
    valid_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()

    if valid_dates.empty:
        return "-", "-"

    latest_dt = valid_dates.max()
    temp = pd.DataFrame({date_col: [latest_dt.strftime("%Y-%m-%d")]})
    temp = add_period_column(temp, date_col, period_option)

    return temp["analysis_period"].iloc[0], temp["analysis_period_label"].iloc[0]



def period_article_summary(df, period_col="analysis_period"):
    if df.empty or period_col not in df.columns:
        return pd.DataFrame(columns=[period_col, "article_count", "prev_count", "change", "change_rate"])

    period_df = (
        df.groupby(period_col)
        .size()
        .reset_index(name="article_count")
        .sort_values(period_col)
    )
    period_df["prev_count"] = period_df["article_count"].shift(1)
    period_df["change"] = period_df["article_count"] - period_df["prev_count"]
    period_df["change_rate"] = (
        period_df["change"] / period_df["prev_count"].replace(0, pd.NA) * 100
    ).round(1)
    period_df[["prev_count", "change", "change_rate"]] = period_df[["prev_count", "change", "change_rate"]].fillna(0)
    return period_df


def period_keyword_movers(df, period_col, keywords, top_n=10):
    periods = sorted(df[period_col].dropna().unique())
    if len(periods) < 2:
        return pd.DataFrame(columns=["keyword", "current_period", "current_count", "prev_period", "prev_count", "change", "change_rate"])

    prev_period, current_period = periods[-2], periods[-1]
    prev_df = df[df[period_col] == prev_period]
    current_df = df[df[period_col] == current_period]

    rows = []
    for kw in keywords:
        prev_count = int(text_series(prev_df).str.contains(kw, case=False, regex=False).sum())
        current_count = int(text_series(current_df).str.contains(kw, case=False, regex=False).sum())
        change = current_count - prev_count
        change_rate = round(change / prev_count * 100, 1) if prev_count else (100.0 if current_count else 0.0)
        if prev_count > 0 or current_count > 0:
            rows.append({
                "keyword": kw,
                "current_period": current_period,
                "current_count": current_count,
                "prev_period": prev_period,
                "prev_count": prev_count,
                "change": change,
                "change_rate": change_rate,
            })

    if not rows:
        return pd.DataFrame(columns=["keyword", "current_period", "current_count", "prev_period", "prev_count", "change", "change_rate"])
    return pd.DataFrame(rows).sort_values(["change", "current_count"], ascending=False).head(top_n)


def period_topic_summary(df, period_col="analysis_period"):
    rows = []
    for period in sorted(df[period_col].dropna().unique()):
        sub = df[df[period_col] == period]
        topic_df = topic_counts(sub)
        if topic_df.empty:
            continue
        for _, row in topic_df.iterrows():
            rows.append({
                "period": period,
                "topic": row["topic"],
                "count": int(row["count"]),
                "keywords": row.get("keywords", "")
            })
    return pd.DataFrame(rows)


# 공통 분석 데이터
article_daily_df = daily_article_summary(df, DATE_COL)
article_top_days = article_daily_df.sort_values("article_count", ascending=False).head(5)
article_surge_days = article_daily_df.sort_values("change", ascending=False).head(5)
all_keyword_df = keyword_counts(df, MAIN_KEYWORDS).head(15)
latest_keyword_df = keyword_counts(latest_df, MAIN_KEYWORDS).head(10)
keyword_movers_df = top_keyword_movers(df, DATE_COL, MAIN_KEYWORDS, top_n=10)
dominant_topic_df = dominant_topic_by_date(df, DATE_COL)

# =========================================================
# Top Tab Menu: 줄이고, 분석 흐름 중심으로 재구성
# =========================================================

tab_overview, tab_keyword, tab_time, tab_topic, tab_risk, tab_source, tab_network, tab_evidence = st.tabs([
    "Executive Summary",
    "Keyword Diagnosis",
    "Time Trend",
    "Topic Shift",
    "Risk Signal",
    "Source Bias",
    "Issue Network",
    "Evidence Table"
])

# =========================================================
# 1. Executive Summary
# =========================================================

with tab_overview:
    st.subheader("Executive Summary")
    st.caption("분석 단위를 선택하면 기사 수, 핵심 키워드, 주제, 리스크 비중이 모두 같은 기준으로 다시 계산됩니다.")

    period_option = st.selectbox(
        "분석 단위 선택",
        ["일별", "주별", "월별", "연별"],
        index=0,
        key="summary_period_option"
    )

    # -----------------------------------------------------
    # 1) 선택한 단위 기준으로 기간 컬럼 생성
    # -----------------------------------------------------
    summary_df = add_period_column(df, DATE_COL, period_option)
    PERIOD_COL = "analysis_period"
    PERIOD_LABEL_COL = "analysis_period_label"

    period_label_map = (
        summary_df[[PERIOD_COL, PERIOD_LABEL_COL]]
        .drop_duplicates()
        .set_index(PERIOD_COL)[PERIOD_LABEL_COL]
        .to_dict()
    )

    periods = sorted([p for p in summary_df[PERIOD_COL].dropna().unique().tolist() if p != "unknown"])
    current_period = periods[-1] if periods else None
    previous_period = periods[-2] if len(periods) >= 2 else None

    current_period_label = period_label_map.get(current_period, current_period or "-")
    previous_period_label = period_label_map.get(previous_period, previous_period or "-")

    # -----------------------------------------------------
    # 2) 가장 중요한 부분: 선택한 현재 구간만 필터링
    #    아래 모든 Summary 지표는 active_df 기준으로 계산
    # -----------------------------------------------------
    if current_period:
        active_df = summary_df[summary_df[PERIOD_COL] == current_period].copy()
    else:
        active_df = summary_df.head(0).copy()

    if previous_period:
        previous_df = summary_df[summary_df[PERIOD_COL] == previous_period].copy()
    else:
        previous_df = summary_df.head(0).copy()

    # -----------------------------------------------------
    # 3) 현재 구간 기준 지표 계산
    # -----------------------------------------------------
    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()
    total_articles = len(summary_df)
    active_articles = len(active_df)
    total_periods = len(periods)

    active_keyword_df = keyword_counts(active_df, MAIN_KEYWORDS).head(10)
    active_topic_df = topic_counts(active_df).head(10)

    active_top_keyword = safe_first_value(active_keyword_df, "keyword")
    active_top_keyword_count = int(safe_first_value(active_keyword_df, "count", 0)) if not active_keyword_df.empty else 0

    active_top_topic = safe_first_value(active_topic_df, "topic")
    active_top_topic_count = int(safe_first_value(active_topic_df, "count", 0)) if not active_topic_df.empty else 0

    risk_keywords = ["보안", "해킹", "개인정보", "랜섬웨어", "침해", "취약점"]
    active_risk_count = len(filter_keywords(active_df, risk_keywords))
    active_risk_ratio = round(active_risk_count / active_articles * 100, 1) if active_articles else 0

    # -----------------------------------------------------
    # 4) 기간별 기사 수/증감 계산
    # -----------------------------------------------------
    period_summary_df = period_article_summary(summary_df, PERIOD_COL)
    if not period_summary_df.empty:
        period_summary_df["period_label"] = period_summary_df[PERIOD_COL].map(period_label_map).fillna(period_summary_df[PERIOD_COL])
    else:
        period_summary_df = pd.DataFrame(columns=[PERIOD_COL, "article_count", "prev_count", "change", "change_rate", "period_label"])

    period_top_df = period_summary_df.sort_values("article_count", ascending=False).head(5)
    period_surge_df = period_summary_df.sort_values("change", ascending=False).head(5)

    period_movers_df = period_keyword_movers(summary_df, PERIOD_COL, MAIN_KEYWORDS, top_n=10)
    if not period_movers_df.empty:
        period_movers_df["current_period_label"] = period_movers_df["current_period"].map(period_label_map).fillna(period_movers_df["current_period"])
        period_movers_df["prev_period_label"] = period_movers_df["prev_period"].map(period_label_map).fillna(period_movers_df["prev_period"])

    period_topic_ts_df = period_topic_summary(summary_df, PERIOD_COL)
    if not period_topic_ts_df.empty:
        period_topic_ts_df["period_label"] = period_topic_ts_df["period"].map(period_label_map).fillna(period_topic_ts_df["period"])

    # -----------------------------------------------------
    # 5) 상단 카드: 모두 active_df 기준
    # -----------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card(
            "현재 분석 구간",
            current_period_label,
            f"원본 수집 범위 {min_date} ~ {max_date} / {period_option} 기준 {total_periods:,}개 구간"
        )
    with c2:
        card(
            "현재 구간 기사 수",
            f"{active_articles:,}건",
            f"전체 수집 기사 {total_articles:,}건 중 현재 {period_option} 구간만 계산"
        )
    with c3:
        card(
            "현재 구간 핵심 키워드",
            active_top_keyword,
            f"{current_period_label} 기준 {active_top_keyword_count:,}건"
        )
    with c4:
        card(
            "현재 구간 리스크 비중",
            f"{active_risk_ratio}%",
            f"보안/개인정보 계열 {active_risk_count:,}건",
            "#ef4444"
        )

    section("현재 선택 구간 기준 핵심 해석")

    peak_period = safe_first_value(period_top_df, "period_label")
    peak_count = int(safe_first_value(period_top_df, "article_count", 0)) if not period_top_df.empty else 0
    top_surge_kw = safe_first_value(period_movers_df, "keyword")
    top_surge_change = int(safe_first_value(period_movers_df, "change", 0)) if not period_movers_df.empty else 0

    render_insight_box(
        f"{period_option} 기준 IT 뉴스 분석 결과",
        [
            f"현재 분석 구간은 '{current_period_label}'이며, 이 구간만 필터링하여 핵심 지표를 다시 계산했습니다.",
            f"현재 구간 기사 수는 {active_articles:,}건입니다.",
            f"현재 구간의 핵심 키워드는 '{active_top_keyword}'이며, {active_top_keyword_count:,}건 확인됩니다.",
            f"현재 구간의 핵심 주제는 '{active_top_topic}'이며, {active_top_topic_count:,}건 관련 기사가 확인됩니다.",
            f"현재 구간 리스크 기사 비중은 {active_risk_ratio}%입니다.",
            f"기사량이 가장 집중된 {period_option} 구간은 {peak_period}이며, 해당 구간 기사 수는 {peak_count:,}건입니다.",
            f"직전 구간({previous_period_label}) 대비 가장 크게 증가한 키워드는 '{top_surge_kw}'이며, 변화량은 {top_surge_change:+,}건입니다.",
        ]
    )

    # 확인용: 실제 필터가 바뀌는지 바로 검증 가능하게 표시
    with st.expander("현재 구간 필터 검증"):
        check_cols = [c for c in [DATE_COL, PERIOD_COL, PERIOD_LABEL_COL, "title"] if c in active_df.columns]
        st.write(f"현재 분석 단위: {period_option}")
        st.write(f"현재 period value: {current_period}")
        st.write(f"현재 period label: {current_period_label}")
        st.write(f"active_df 기사 수: {len(active_df):,}건")
        if check_cols:
            st.dataframe(active_df[check_cols].head(20), use_container_width=True)

    section("분석 단위별 기사량 변화")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"### {period_option} 기사량 추이")
        if not period_summary_df.empty:
            st.line_chart(period_summary_df.set_index("period_label")[["article_count"]])
            st.dataframe(period_summary_df.sort_values(PERIOD_COL, ascending=False), use_container_width=True)
        else:
            st.warning("기사량 집계 데이터가 없습니다.")
    with col_b:
        progress_list(period_top_df, "period_label", "article_count", f"기사량이 많은 {period_option} 구간 TOP 5", top_n=5)
        st.markdown("### 증가폭 TOP 5")
        st.dataframe(period_surge_df, use_container_width=True)

    section(f"현재 {period_option} 구간 기준 키워드/주제 결과")
    st.caption("아래 키워드와 주제는 전체 기간이 아니라 현재 선택 구간(active_df) 기준으로만 계산됩니다.")

    col_c, col_d = st.columns(2)
    with col_c:
        progress_list(active_keyword_df, "keyword", "count", f"현재 구간({current_period_label}) 키워드 TOP 10", top_n=10)
        st.dataframe(active_keyword_df, use_container_width=True)
    with col_d:
        progress_list(active_topic_df, "topic", "count", f"현재 구간({current_period_label}) 주제 TOP", top_n=10)
        st.dataframe(active_topic_df, use_container_width=True)

    section("직전 구간 대비 키워드 변화")
    st.caption(f"{previous_period_label} → {current_period_label} 기준으로 키워드 증가/감소를 계산합니다.")
    if period_movers_df.empty:
        st.warning("비교 가능한 이전 구간이 부족합니다. 데이터가 더 쌓이면 변화 분석이 가능합니다.")
    else:
        st.dataframe(period_movers_df, use_container_width=True)
        progress_list(period_movers_df, "keyword", "change", "증가 키워드 TOP 10", top_n=10, suffix="건")

    section("주제별 구간 변화")
    if period_topic_ts_df.empty:
        st.warning("주제별 구간 변화 데이터가 없습니다.")
    else:
        st.dataframe(period_topic_ts_df, use_container_width=True)
        selected_period_topic = st.selectbox(
            "상세 확인 주제",
            sorted(period_topic_ts_df["topic"].dropna().unique().tolist()),
            key="summary_period_topic_select"
        )
        st.dataframe(
            period_topic_ts_df[period_topic_ts_df["topic"] == selected_period_topic].sort_values("period", ascending=False),
            use_container_width=True
        )

    st.info(f"""
    현재 Summary는 '{period_option}' 기준으로 최신 구간({current_period_label})만 active_df로 필터링한 뒤,
    기사 수, 키워드 순위, 주제 분포, 리스크 비중을 모두 active_df 기준으로 재계산합니다.
    """)


# =========================================================
# 2. Keyword Diagnosis
# =========================================================

with tab_keyword:
    st.subheader("Keyword Diagnosis")
    st.caption("키워드별 단순 언급량이 아니라, 등장 지속성·집중일·최근 변화·트렌드 유형을 함께 진단합니다.")

    keyword_options = all_keyword_df["keyword"].tolist() if not all_keyword_df.empty else MAIN_KEYWORDS
    selected_kw = st.selectbox("분석할 키워드 선택", keyword_options, key="keyword_diagnosis_select")

    profile = keyword_profile(df, DATE_COL, selected_kw)
    selected_daily = keyword_daily_detail(df, DATE_COL, selected_kw)
    selected_articles = filter_keyword(df, selected_kw)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("전체 언급량", f"{profile['total']:,}건", f"전체 기간 기준")
    with c2:
        card("등장 일수", f"{profile['active_days']:,}일", f"총 {df[DATE_COL].nunique():,}일 중")
    with c3:
        card("최고 집중일", profile["peak_date"], f"{profile['peak_count']:,}건")
    with c4:
        color = "#ef4444" if profile["trend_type"] == "급등형" else "#22c55e" if profile["trend_type"] == "지속형" else "#eab308" if profile["trend_type"] == "이벤트형" else "#38bdf8"
        card("트렌드 유형", profile["trend_type"], f"최근 변화 {profile['latest_change']:+,}건", color)

    section("선택 키워드 해석")
    render_insight_box(
        f"'{selected_kw}' 키워드 진단",
        [
            f"'{selected_kw}'는 전체 기간 {profile['total']:,}건 등장했고, {profile['active_days']:,}일 동안 반복적으로 관찰되었습니다.",
            f"가장 강하게 나타난 날짜는 {profile['peak_date']}이며, 해당일 {profile['peak_count']:,}건이 확인됩니다.",
            f"최신일 기준 언급량은 {profile['latest_count']:,}건이고, 전일 대비 {profile['latest_change']:+,}건 변화했습니다.",
            f"따라서 이 키워드는 현재 '{profile['trend_type']}' 성격으로 분류할 수 있습니다.",
        ]
    )

    section("키워드 변화 추이")
    st.line_chart(selected_daily.set_index("date")[["count"]])
    st.dataframe(selected_daily.sort_values("date", ascending=False), use_container_width=True)

    section("핵심 키워드 중요도")
    st.caption("미리 정의한 IT 후보 키워드 안에서 기사 수와 희소성을 함께 반영한 중요도입니다.")
    tfidf_df = tfidf_keywords(df, 20)
    st.dataframe(tfidf_df, use_container_width=True)

    section("관련 기사 근거")
    compact_article_evidence(selected_articles, top_n=10)


# =========================================================
# 3. Time Trend
# =========================================================

with tab_time:
    st.subheader("Time Trend")
    st.caption("날짜별 기사량과 키워드 변화를 통해 언제 이슈가 집중되었는지 분석합니다.")

    c1, c2, c3 = st.columns(3)
    with c1:
        avg_articles = round(article_daily_df["article_count"].mean(), 1) if not article_daily_df.empty else 0
        card("일평균 기사 수", f"{avg_articles:,}건", "전체 기간 평균")
    with c2:
        peak_day = safe_first_value(article_top_days, "date")
        peak_count = int(safe_first_value(article_top_days, "article_count", 0)) if not article_top_days.empty else 0
        card("최대 기사량 날짜", peak_day, f"{peak_count:,}건")
    with c3:
        surge_day = safe_first_value(article_surge_days, "date")
        surge_change = int(safe_first_value(article_surge_days, "change", 0)) if not article_surge_days.empty else 0
        card("최대 증가 날짜", surge_day, f"전일 대비 {surge_change:+,}건", "#ef4444")

    section("일별 기사량 추이")
    if not article_daily_df.empty:
        st.line_chart(article_daily_df.set_index("date")[["article_count"]])
        st.dataframe(article_daily_df.sort_values("date", ascending=False), use_container_width=True)
    else:
        st.warning("일별 기사량 데이터가 없습니다.")

    section("날짜별 핵심 키워드 변화")
    render_daily_keyword_timeline(daily_kw, top_n=5)

    section("기사량 급증일 해석")
    if not article_surge_days.empty:
        selected_date = st.selectbox("급증일 관련 기사 확인", article_surge_days["date"].tolist(), key="surge_date_select")
        day_df = df[df[DATE_COL] == selected_date]
        day_keywords = keyword_counts(day_df, MAIN_KEYWORDS).head(8)
        progress_list(day_keywords, "keyword", "count", f"{selected_date} 주요 키워드", top_n=8)
        compact_article_evidence(day_df, top_n=10)


# =========================================================
# 4. Topic Shift
# =========================================================

with tab_topic:
    st.subheader("Topic Shift")
    st.caption("AI·반도체·보안·클라우드 등 주제군 비중이 기간별로 어떻게 바뀌었는지 분석합니다.")

    topic_all = topic_counts(df)
    topic_latest = topic_counts(latest_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        card("전체 1위 주제", safe_first_value(topic_all, "topic"), f"{int(safe_first_value(topic_all, 'count', 0)):,}건")
    with c2:
        card("최신일 1위 주제", safe_first_value(topic_latest, "topic"), f"{int(safe_first_value(topic_latest, 'count', 0)):,}건")
    with c3:
        risk_topic = topic_sent_df.sort_values("risk_ratio", ascending=False).iloc[0] if not topic_sent_df.empty else None
        card("리스크 비중 높은 주제", risk_topic["topic"] if risk_topic is not None else "-", f"{risk_topic['risk_ratio']}%" if risk_topic is not None else "-", "#ef4444")

    section("주제별 전체 비중")
    progress_list(topic_all, "topic", "count", "전체 기간 주제 분포", top_n=8)

    section("주제별 일자 변화")
    st.dataframe(topic_ts_df, use_container_width=True)
    selected_topic = st.selectbox("상세 확인 주제", list(TOPIC_MAP.keys()), key="topic_shift_select")
    topic_detail = topic_ts_df[["date", selected_topic]].sort_values("date")
    st.line_chart(topic_detail.set_index("date")[[selected_topic]])
    st.dataframe(topic_detail.sort_values(selected_topic, ascending=False), use_container_width=True)

    section("날짜별 지배 주제")
    st.caption("각 날짜에 가장 많이 보도된 주제군을 요약합니다.")
    st.dataframe(dominant_topic_df, use_container_width=True)


# =========================================================
# 5. Risk Signal
# =========================================================

with tab_risk:
    st.subheader("Risk Signal")
    st.caption("보안·해킹·개인정보 등 부정/리스크 신호를 따로 분리해 확인합니다.")

    risk_df = filter_keywords(df, risk_keywords)
    risk_ratio = round(len(risk_df) / len(df) * 100, 1) if len(df) else 0

    latest_risk_df = filter_keywords(latest_df, risk_keywords)
    neg_df = df[df["sentiment_group"] == "부정/리스크"]

    c1, c2, c3 = st.columns(3)
    with c1:
        card("리스크 키워드 기사", f"{len(risk_df):,}건", "전체 대비 " + str(risk_ratio) + "%", "#ef4444")
    with c2:
        card("최신일 리스크 기사", f"{len(latest_risk_df):,}건", f"{latest_date} 기준", "#ef4444")
    with c3:
        neg_ratio = round(len(neg_df) / len(df) * 100, 1) if len(df) else 0
        card("부정/리스크 분류", f"{neg_ratio}%", f"{len(neg_df):,}건", "#ef4444")

    section("주제별 리스크 비율")
    st.dataframe(topic_sent_df.sort_values("risk_ratio", ascending=False), use_container_width=True)

    section("리스크 키워드 TOP")
    risk_keyword_df = keyword_counts(risk_df, risk_keywords)
    progress_list(risk_keyword_df, "keyword", "count", "리스크 키워드 빈도", top_n=10)

    section("리스크 기사 근거")
    compact_article_evidence(risk_df, top_n=15)


# =========================================================
# 6. Source Bias
# =========================================================

with tab_source:
    st.subheader("Source Bias")
    st.caption("언론사/수집 출처별로 어떤 주제와 프레임이 강하게 나타나는지 비교합니다.")

    source_count = df["analysis_source"].value_counts().reset_index()
    source_count.columns = ["source", "count"]

    c1, c2, c3 = st.columns(3)
    with c1:
        card("수집 소스 수", f"{df['analysis_source'].nunique():,}개", "processed 데이터 기준")
    with c2:
        card("최다 기사 소스", safe_first_value(source_count, "source"), f"{int(safe_first_value(source_count, 'count', 0)):,}건")
    with c3:
        top_source_ratio = round(int(safe_first_value(source_count, "count", 0)) / len(df) * 100, 1) if len(df) else 0
        card("상위 소스 집중도", f"{top_source_ratio}%", "전체 기사 중 1위 소스 비중")

    section("소스별 기사량")
    progress_list(source_count, "source", "count", "수집 소스별 기사 수", top_n=12)

    section("소스별 주제 비중")
    st.dataframe(source_topic_ratio(df), use_container_width=True)

    section("소스별 보도 프레임")
    frame_df = source_frame(df)
    st.dataframe(frame_df, use_container_width=True)

    selected_source = st.selectbox("소스별 기사 근거 확인", source_count["source"].tolist(), key="source_bias_select")
    selected_source_df = df[df["analysis_source"] == selected_source]
    progress_list(keyword_counts(selected_source_df, MAIN_KEYWORDS).head(8), "keyword", "count", f"{selected_source} 주요 키워드", top_n=8)
    compact_article_evidence(selected_source_df, top_n=10)


# =========================================================
# 7. Issue Network
# =========================================================

with tab_network:
    st.subheader("Issue Network")
    st.caption("같은 기사 안에서 함께 등장한 키워드 쌍을 분석해 이슈 간 연결 구조를 보여줍니다.")

    render_network_summary(net_df)

    section("키워드 연결 그래프")
    if not net_df.empty:
        components.html(network_html(net_df), height=760, scrolling=True)
    else:
        st.warning("네트워크 그래프를 생성할 수 있는 데이터가 없습니다.")

    section("강한 연결 관계")
    render_strong_connections(net_df, top_n=6)

    section("연결 데이터 상세")
    st.dataframe(net_df.head(30), use_container_width=True)


# =========================================================
# 8. Evidence Table
# =========================================================

with tab_evidence:
    st.subheader("Evidence Table")
    st.caption("분석 결과를 뒷받침하는 기사 원문 데이터입니다. 발표 자료에 넣을 근거를 찾는 용도입니다.")

    search_kw = st.text_input("기사 검색 키워드", "")
    source_options = ["전체"] + sorted(df["analysis_source"].dropna().unique().tolist())
    source_sel = st.selectbox("소스 필터", source_options)

    view = df.copy()
    if search_kw.strip():
        view = filter_keyword(view, search_kw.strip())
    if source_sel != "전체":
        view = view[view["analysis_source"] == source_sel]

    c1, c2 = st.columns(2)
    with c1:
        card("검색 결과", f"{len(view):,}건", "현재 필터 기준")
    with c2:
        card("전체 데이터", f"{len(df):,}건", "processed CSV 기준")

    article_table(view, DATE_COL)
