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
# Sidebar Menu
# =========================================================

# =========================================================
# Top Tab Menu
# =========================================================

tab_home, tab_summary, tab_keyword, tab_trend, tab_surge, tab_source, tab_sentiment, tab_network = st.tabs([
    "Home",
    "트렌드 요약",
    "IT 키워드 동향",
    "시기별 트렌드",
    "급등/지속 트렌드",
    "출처별 프레임",
    "리스크 분석",
    "네트워크 분석"
])

# =========================================================
# Pages
# =========================================================


with tab_home:
    st.subheader("오늘의 IT 뉴스 브리핑")

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
        card("오늘의 연결 이슈", mega_trend, mega_desc)
    with c4:
        card("오늘 리스크 이슈", "보안 / 개인정보", f"{risk_today_count:,}건 탐지", "#ef4444")


    section("오늘의 IT 키워드 TOP 10")
    keyword_chip_grid(top_keywords, "keyword", "count", None, clickable=True, session_key="home_drill_keyword")

    if "home_drill_keyword" in st.session_state and st.session_state["home_drill_keyword"]:
        drill_kw = st.session_state["home_drill_keyword"]
        drill_df = filter_keyword(df, drill_kw)
        st.markdown("### 선택 키워드 관련 뉴스")
        st.info(f"선택된 키워드: {drill_kw} / 관련 기사 수: {len(drill_df):,}건")
        article_table(drill_df, DATE_COL)

    section("오늘의 핵심 인사이트")

    ai_semi_count = 0
    risk_count = 0
    cloud_count = 0

    if not latest_df.empty:
        ai_semi_count = len(filter_keywords(latest_df, ["AI반도체", "AI 반도체", "HBM", "GPU", "엔비디아", "반도체"]))
        risk_count = len(filter_keywords(latest_df, ["보안", "해킹", "개인정보", "랜섬웨어", "침해", "취약점"]))
        cloud_count = len(filter_keywords(latest_df, ["클라우드", "AWS", "Azure", "데이터센터", "서버", "인프라"]))

    insight_cards = [
        {
            "title": "AI와 반도체 결합 이슈",
            "value": f"{ai_semi_count:,}건",
            "desc": "HBM · GPU · AI반도체 관련 보도가 함께 증가하는 흐름",
            "color": "#38bdf8"
        },
        {
            "title": "보안 리스크 이슈",
            "value": f"{risk_count:,}건",
            "desc": "개인정보 · 해킹 · 침해 · 취약점 중심의 리스크 보도 감지",
            "color": "#ef4444"
        },
        {
            "title": "클라우드 인프라 이슈",
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


with tab_summary:
    st.subheader("트렌드 요약")
    st.caption("전체 수집 기간의 IT 뉴스 흐름을 핵심 키워드, 지속/급등 트렌드, 기사량 급증일, 자동 해석 인사이트 기준으로 요약합니다.")

    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()
    total_days = df[DATE_COL].nunique()

    daily_article_count = (
        df.groupby(DATE_COL)
        .size()
        .reset_index(name="article_count")
        .sort_values("article_count", ascending=False)
    )

    all_keyword_df = keyword_counts(df, MAIN_KEYWORDS).head(10)

    top_day = daily_article_count.iloc[0][DATE_COL] if not daily_article_count.empty else "-"
    top_day_count = int(daily_article_count.iloc[0]["article_count"]) if not daily_article_count.empty else 0

    top_keyword = all_keyword_df.iloc[0]["keyword"] if not all_keyword_df.empty else "-"
    top_keyword_count = int(all_keyword_df.iloc[0]["count"]) if not all_keyword_df.empty else 0

    if not surge_df.empty:
        sustained_df = surge_df[surge_df["trend_type"] == "지속형"].sort_values("total_count", ascending=False)
        surged_df = surge_df[surge_df["trend_type"] == "급등형"].sort_values("max_growth_rate", ascending=False)
        event_df = surge_df[surge_df["trend_type"] == "이벤트형"].sort_values("max_count", ascending=False)
        observation_df = surge_df[surge_df["trend_type"] == "관찰형"].sort_values("total_count", ascending=False)
    else:
        sustained_df = pd.DataFrame()
        surged_df = pd.DataFrame()
        event_df = pd.DataFrame()
        observation_df = pd.DataFrame()

    sustained_kw = sustained_df.iloc[0]["keyword"] if not sustained_df.empty else "-"
    surged_kw = surged_df.iloc[0]["keyword"] if not surged_df.empty else "-"
    event_kw = event_df.iloc[0]["keyword"] if not event_df.empty else "-"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("분석 기간", f"{min_date} ~ {max_date}", f"총 {total_days:,}일 기준")
    with c2:
        card("전체 기사 수", f"{len(df):,}건", "S3 processed CSV 기준")
    with c3:
        card("최다 언급 키워드", top_keyword, f"{top_keyword_count:,}건 언급")
    with c4:
        card("기사량 최다 날짜", top_day, f"{top_day_count:,}건 수집")

    section("전체 기간 핵심 키워드 TOP 10", "전체 수집 기간에서 반복적으로 많이 등장한 IT 키워드입니다.")
    keyword_chip_grid(all_keyword_df, "keyword", "count", None)

    section("트렌드 유형별 핵심 키워드", "단순 빈도가 아니라 등장 일수와 증가율을 함께 반영해 키워드 성격을 분류합니다.")

    t1, t2, t3 = st.columns(3)

    with t1:
        card(
            "지속형 대표 키워드",
            sustained_kw,
            "여러 날짜에 반복적으로 등장한 장기 관심 이슈",
            "#22c55e"
        )
        if not sustained_df.empty:
            progress_list(
                sustained_df.head(5),
                "keyword",
                "total_count",
                "지속형 키워드 TOP 5",
                top_n=5
            )
        else:
            st.warning("지속형 키워드가 없습니다.")

    with t2:
        card(
            "급등형 대표 키워드",
            surged_kw,
            "전일 대비 증가율이 크게 나타난 단기 이슈",
            "#ef4444"
        )
        if not surged_df.empty:
            progress_list(
                surged_df.head(5),
                "keyword",
                "max_growth_rate",
                "급등형 키워드 TOP 5",
                top_n=5,
                suffix="%"
            )
        else:
            st.warning("급등형 키워드가 없습니다.")

    with t3:
        card(
            "이벤트형 대표 키워드",
            event_kw,
            "특정 날짜에 집중적으로 등장한 이벤트성 이슈",
            "#eab308"
        )
        if not event_df.empty:
            progress_list(
                event_df.head(5),
                "keyword",
                "max_count",
                "이벤트형 키워드 TOP 5",
                top_n=5
            )
        else:
            st.warning("이벤트형 키워드가 없습니다.")

    section("기사량 급증 날짜 TOP 5", "수집 기간 중 기사량이 가장 많았던 날짜를 기준으로 주요 이슈 발생 구간을 확인합니다.")

    if not daily_article_count.empty:
        progress_list(
            daily_article_count.head(5),
            DATE_COL,
            "article_count",
            "기사량이 많았던 날짜",
            top_n=5
        )
        st.dataframe(daily_article_count.head(10), use_container_width=True)
    else:
        st.warning("일별 기사량 데이터가 없습니다.")

    section("주제별 흐름 요약", "AI, 반도체, 클라우드, 보안 등 주요 주제가 기간 중 어떻게 분포했는지 확인합니다.")

    topic_total_df = topic_counts(df)
    if not topic_total_df.empty:
        progress_list(topic_total_df, "topic", "count", "전체 기간 주제별 기사 수", top_n=6)
        st.dataframe(topic_total_df, use_container_width=True)
    else:
        st.warning("주제별 분석 데이터가 없습니다.")

    section("자동 해석 인사이트")

    if not all_keyword_df.empty:
        top3_keywords = ", ".join(all_keyword_df.head(3)["keyword"].astype(str).tolist())
    else:
        top3_keywords = "-"

    if not topic_total_df.empty:
        top_topic = topic_total_df.iloc[0]["topic"]
        top_topic_count = int(topic_total_df.iloc[0]["count"])
    else:
        top_topic = "-"
        top_topic_count = 0

    insight_1 = f"전체 기간 중 가장 많이 언급된 키워드는 '{top_keyword}'이며, 총 {top_keyword_count:,}건의 기사에서 확인되었습니다. 주요 상위 키워드는 {top3_keywords}입니다."
    insight_2 = f"기사량이 가장 많았던 날짜는 {top_day}로, 해당 날짜에는 총 {top_day_count:,}건의 기사가 수집되었습니다."
    insight_3 = f"주제 기준으로는 '{top_topic}' 관련 기사가 {top_topic_count:,}건으로 가장 높게 나타났습니다."
    insight_4 = f"트렌드 유형 기준으로는 지속형 '{sustained_kw}', 급등형 '{surged_kw}', 이벤트형 '{event_kw}'가 대표적으로 나타났습니다."

    st.info(
        f"""
        {insight_1}

        {insight_2}

        {insight_3}

        {insight_4}

        따라서 이번 수집 기간의 IT 뉴스 흐름은 단순한 기사량 집계가 아니라, 반복적으로 등장한 장기 이슈와 특정 시점에 급증한 이벤트성 이슈가 함께 나타난 구조로 해석할 수 있습니다.
        """
    )

    section("트렌드 상세 데이터")

    detail_tabs = st.tabs(["지속형", "급등형", "이벤트형", "관찰형", "전체"])

    with detail_tabs[0]:
        if sustained_df.empty:
            st.warning("지속형 데이터가 없습니다.")
        else:
            st.dataframe(sustained_df, use_container_width=True)

    with detail_tabs[1]:
        if surged_df.empty:
            st.warning("급등형 데이터가 없습니다.")
        else:
            st.dataframe(surged_df, use_container_width=True)

    with detail_tabs[2]:
        if event_df.empty:
            st.warning("이벤트형 데이터가 없습니다.")
        else:
            st.dataframe(event_df, use_container_width=True)

    with detail_tabs[3]:
        if observation_df.empty:
            st.warning("관찰형 데이터가 없습니다.")
        else:
            st.dataframe(observation_df, use_container_width=True)

    with detail_tabs[4]:
        if surge_df.empty:
            st.warning("급등/지속 트렌드 분석 데이터가 없습니다.")
        else:
            st.dataframe(surge_df, use_container_width=True)


with tab_keyword:
    st.subheader("IT 키워드 동향 분석")
    st.caption("단순 키워드 검색이 아니라, 전체 기간과 최신일의 키워드 변화·집중도·트렌드 유형을 함께 분석합니다.")

    # -----------------------------------------------------
    # 1) 전체 기간 vs 최신일 키워드 비교
    # -----------------------------------------------------
    section("핵심 키워드 순위 비교", "전체 기간과 최신일 기준 키워드 순위를 비교해 현재 이슈가 누적형인지, 최근 급부상형인지 확인합니다.")

    all_keyword_rank = keyword_counts(df, MAIN_KEYWORDS).head(15)
    latest_keyword_rank = keyword_counts(latest_df, MAIN_KEYWORDS).head(15)

    rank_col1, rank_col2 = st.columns(2)

    with rank_col1:
        keyword_chip_grid(
            all_keyword_rank.head(10),
            "keyword",
            "count",
            "전체 기간 핵심 키워드 TOP 10",
            top_n=10
        )

    with rank_col2:
        keyword_chip_grid(
            latest_keyword_rank.head(10),
            "keyword",
            "count",
            f"최신일 핵심 키워드 TOP 10 ({latest_date})",
            top_n=10
        )

    # -----------------------------------------------------
    # 2) 키워드 선택
    # -----------------------------------------------------
    keyword_options = all_keyword_rank["keyword"].tolist() if not all_keyword_rank.empty else MAIN_KEYWORDS
    selected_kw = st.selectbox("심층 분석할 키워드 선택", keyword_options, key="keyword_page_select")

    selected_articles = filter_keyword(df, selected_kw)
    latest_selected = filter_keyword(latest_df, selected_kw)

    # 선택 키워드 일별 집계
    selected_daily = (
        selected_articles.groupby(DATE_COL)
        .size()
        .reset_index(name="count")
        .sort_values(DATE_COL)
    ) if not selected_articles.empty else pd.DataFrame(columns=[DATE_COL, "count"])

    selected_total = len(selected_articles)
    selected_today = len(latest_selected)
    selected_active_days = selected_daily[DATE_COL].nunique() if not selected_daily.empty else 0

    if not selected_daily.empty:
        selected_peak_row = selected_daily.sort_values("count", ascending=False).iloc[0]
        selected_peak_date = selected_peak_row[DATE_COL]
        selected_peak_count = int(selected_peak_row["count"])
    else:
        selected_peak_date = "-"
        selected_peak_count = 0

    # 전일 대비 변화량 계산
    sorted_dates = sorted(df[DATE_COL].dropna().unique())
    prev_date = None
    if latest_date in sorted_dates:
        latest_idx = sorted_dates.index(latest_date)
        if latest_idx > 0:
            prev_date = sorted_dates[latest_idx - 1]

    prev_count = 0
    if prev_date:
        prev_count = len(filter_keyword(df[df[DATE_COL] == prev_date], selected_kw))

    diff_count = selected_today - prev_count
    if prev_count > 0:
        diff_rate = round(diff_count / prev_count * 100, 1)
        diff_text = f"{diff_count:+,}건 / {diff_rate:+.1f}%"
    elif selected_today > 0:
        diff_text = "NEW"
    else:
        diff_text = "0건"

    # 트렌드 유형 가져오기
    trend_type = "관찰형"
    trend_reason = "선택 키워드가 전체 기간에서 제한적으로 관찰되는 흐름입니다."
    if not surge_df.empty and selected_kw in surge_df["keyword"].values:
        trend_row = surge_df[surge_df["keyword"] == selected_kw].iloc[0]
        trend_type = trend_row["trend_type"]
        if trend_type == "지속형":
            trend_reason = "여러 날짜에 반복적으로 등장해 장기 관심 이슈로 볼 수 있습니다."
        elif trend_type == "급등형":
            trend_reason = "특정 시점에서 전일 대비 증가율이 크게 나타난 이슈입니다."
        elif trend_type == "이벤트형":
            trend_reason = "짧은 기간에 집중적으로 등장해 행사·발표·사고성 이슈일 가능성이 있습니다."
        else:
            trend_reason = "뚜렷한 급등이나 장기 지속보다는 관찰 가능한 보조 이슈입니다."

    # 관련 주제 추정
    related_topics = []
    for topic, kws in TOPIC_MAP.items():
        if selected_kw in kws or len(filter_keywords(selected_articles, kws)) > 0:
            related_topics.append(topic)
    related_topic_text = ", ".join(related_topics[:3]) if related_topics else "직접 매칭 주제 없음"

    # -----------------------------------------------------
    # 3) 선택 키워드 심층 카드
    # -----------------------------------------------------
    section(f"'{selected_kw}' 키워드 심층 분석", "선택한 키워드의 전체 언급량, 최신 변화, 집중 날짜, 트렌드 유형을 요약합니다.")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        card("전체 언급량", f"{selected_total:,}건", "전체 분석 기간 기준")
    with k2:
        card("최신일 언급량", f"{selected_today:,}건", f"{latest_date} 기준")
    with k3:
        card("등장 일수", f"{selected_active_days:,}일", "해당 키워드가 등장한 날짜 수")
    with k4:
        card("최고 집중일", str(selected_peak_date), f"{selected_peak_count:,}건 언급", "#eab308")

    k5, k6, k7 = st.columns(3)
    with k5:
        card("전일 대비 변화", diff_text, f"비교 기준: {prev_date if prev_date else '전일 데이터 없음'}", "#ef4444" if diff_count > 0 else "#38bdf8")
    with k6:
        card("트렌드 유형", trend_type, trend_reason)
    with k7:
        card("관련 주제", related_topic_text, "TOPIC_MAP 기준 연결 주제")

    # -----------------------------------------------------
    # 4) 선택 키워드 일별 추이
    # -----------------------------------------------------
    section(f"'{selected_kw}' 일별 언급 추이", "선택 키워드가 어느 날짜에 집중되었는지 확인합니다.")

    if selected_daily.empty:
        st.warning("선택한 키워드의 일별 데이터가 없습니다.")
    else:
        chart_df = selected_daily.rename(columns={DATE_COL: "date"}).set_index("date")
        st.line_chart(chart_df["count"])

        progress_list(
            selected_daily.sort_values("count", ascending=False).head(10),
            DATE_COL,
            "count",
            f"'{selected_kw}' 언급량이 높았던 날짜 TOP 10",
            top_n=10
        )

    # -----------------------------------------------------
    # 5) 데이터 기반 키워드 중요도
    # -----------------------------------------------------
    section("데이터 기반 핵심 키워드 중요도", "미리 정의한 IT 후보 키워드 안에서 기사 등장 빈도와 중요도를 함께 계산합니다.")

    tfidf_df = tfidf_keywords(df, 25)
    if tfidf_df.empty:
        st.warning("키워드 중요도 데이터를 생성할 수 없습니다.")
    else:
        t1, t2 = st.columns([1, 1])
        with t1:
            progress_list(
                tfidf_df.head(10),
                "keyword",
                "article_count",
                "기사 등장 수 기준 TOP 10",
                top_n=10
            )
        with t2:
            st.dataframe(tfidf_df, use_container_width=True)

    # -----------------------------------------------------
    # 6) 자동 해석 인사이트
    # -----------------------------------------------------
    section("자동 해석 인사이트")

    if selected_total == 0:
        insight_text = f"'{selected_kw}' 키워드는 현재 분석 기간에서 뚜렷하게 확인되지 않았습니다."
    else:
        if trend_type == "지속형":
            trend_comment = "기간 전반에 걸쳐 꾸준히 등장했기 때문에, 일시적 이슈보다는 지속 관심 주제로 해석할 수 있습니다."
        elif trend_type == "급등형":
            trend_comment = "특정 날짜를 기준으로 언급량이 크게 증가했기 때문에, 발표·사건·정책 변화 등 외부 이벤트와 연결해 해석할 필요가 있습니다."
        elif trend_type == "이벤트형":
            trend_comment = "등장 기간은 짧지만 특정 날짜에 집중되어, 단발성 이슈나 행사성 이슈일 가능성이 있습니다."
        else:
            trend_comment = "전체 기간에서 보조적으로 관찰되는 키워드이며, 다른 주요 키워드와 함께 해석하는 것이 적절합니다."

        insight_text = f"""
        '{selected_kw}' 키워드는 전체 기간 동안 총 {selected_total:,}건 등장했으며, {selected_active_days:,}일 동안 관찰되었습니다.

        가장 집중된 날짜는 {selected_peak_date}로, 해당 날짜에 {selected_peak_count:,}건이 확인되었습니다.

        최신일 기준 언급량은 {selected_today:,}건이며, 전일 대비 변화는 {diff_text}입니다.

        이 키워드는 '{trend_type}' 흐름으로 분류되며, {trend_comment}
        """

    st.info(insight_text)

    # -----------------------------------------------------
    # 7) 관련 기사 근거
    # -----------------------------------------------------
    section(f"'{selected_kw}' 관련 기사 근거", "키워드 분석의 근거가 되는 실제 기사 목록입니다.")
    article_table(selected_articles, DATE_COL)

    with st.expander(f"'{selected_kw}'와 유사도가 높은 기사 보기"):
        st.caption("TF-IDF 벡터 간 코사인 유사도 기준으로 선택 키워드와 가까운 기사를 확인합니다.")
        st.dataframe(similar_articles(df, selected_kw), use_container_width=True)

with tab_trend:
    st.subheader("시기별 IT 뉴스 트렌드 분석")
    st.info(f"본 대시보드는 {df[DATE_COL].min()}부터 {df[DATE_COL].max()}까지의 데이터를 기준으로 일별 단위의 IT 뉴스 트렌드를 분석합니다.")
    st.caption("숙제 주제에 맞춰 날짜별 키워드 변화, 주제별 기사량 변화, 기사 급증 구간을 기준으로 시기별 IT 뉴스 트렌드를 분석합니다.")

    render_daily_keyword_timeline(daily_kw, top_n=5)

    section("키워드별 일자 추이", "선택한 키워드가 어느 날짜에 강하게 등장했는지 확인합니다.")
    available_keywords = daily_kw["keyword"].dropna().unique().tolist() if not daily_kw.empty else MAIN_KEYWORDS
    sel_kw = st.selectbox("키워드별 일자 추이 확인", available_keywords)
    render_keyword_trend_detail(daily_kw, sel_kw)

    section("시기별 주제 트렌드", "날짜별로 어떤 IT 주제가 많이 보도되었는지 확인합니다.")
    st.dataframe(topic_ts_df, use_container_width=True)
    sel_topic = st.selectbox("시계열 상세 확인 주제", list(TOPIC_MAP.keys()), key="timeseries_topic")
    st.dataframe(topic_ts_df[["date", sel_topic]].sort_values(sel_topic, ascending=False), use_container_width=True)

    section("시기별 주요 이벤트 구간", "기사량이 급증한 날짜를 기준으로 주요 이슈 구간을 확인합니다.")
    st.dataframe(event_annotations(df, DATE_COL), use_container_width=True)


with tab_surge:
    st.subheader("급등/지속 트렌드 분석")
    st.caption("단순 빈도뿐 아니라 증가율과 등장 일수를 함께 분석하여 키워드를 급등형·지속형·이벤트형으로 분류합니다.")

    if surge_df.empty:
        st.warning("급등/지속 트렌드 분석 데이터가 없습니다.")
    else:
        sustained = surge_df[surge_df["trend_type"] == "지속형"]
        surged = surge_df[surge_df["trend_type"] == "급등형"]
        event_like = surge_df[surge_df["trend_type"] == "이벤트형"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            card("지속형 키워드", f"{len(sustained):,}개", "여러 날짜에 반복적으로 등장")
        with c2:
            card("급등형 키워드", f"{len(surged):,}개", "전일 대비 증가율이 큰 키워드", "#ef4444")
        with c3:
            card("이벤트형 키워드", f"{len(event_like):,}개", "특정 날짜에 집중된 키워드", "#eab308")
        with c4:
            top_total = surge_df.sort_values("total_count", ascending=False).iloc[0]["keyword"] if not surge_df.empty else "-"
            card("대표 트렌드 키워드", top_total, "총 언급량 기준")

        section("트렌드 유형별 상세 결과")
        selected_type = st.selectbox("트렌드 유형 선택", ["전체", "지속형", "급등형", "이벤트형", "관찰형"])

        if selected_type == "전체":
            view = surge_df
        else:
            view = surge_df[surge_df["trend_type"] == selected_type]

        st.dataframe(view, use_container_width=True)

        section("급등률 TOP 10")
        growth_top = surge_df.sort_values("max_growth_rate", ascending=False).head(10)
        progress_list(growth_top, "keyword", "max_growth_rate", "전일 대비 최대 증가율", top_n=10, suffix="%")


with tab_network:
    section("네트워크 분석", "같은 기사 안에서 함께 등장한 키워드 쌍을 분석해 IT 이슈 간 연결 구조를 시각화합니다.")

    render_network_summary(net_df)

    if not net_df.empty:
        components.html(network_html(net_df), height=760, scrolling=True)
    else:
        st.warning("네트워크 그래프를 생성할 수 있는 데이터가 없습니다.")

    render_strong_connections(net_df, top_n=6)

with tab_source:
    section("출처별 보도 프레임 분석", "뉴스 출처별 기사 수, 주요 키워드, 주제 비중, 보도 프레임 차이를 분석합니다.")
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

    section("소스별 보도 프레임")
    frame_df = source_frame(df)
    st.dataframe(frame_df, use_container_width=True)
    frame = st.selectbox("프레임 선택", ["성장/혁신", "보안/리스크", "산업/경쟁", "정책/규제", "인프라/클라우드"])
    if frame in frame_df.columns:
        progress_list(frame_df[["source", frame]].sort_values(frame, ascending=False), "source", frame, f"{frame} 프레임 TOP 10")

with tab_sentiment:
    section("주제별 감성/리스크 교차 분석")
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
