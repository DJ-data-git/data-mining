import streamlit as st
import pandas as pd
import boto3
import re
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile


# ---------------------------
# Page Config
# ---------------------------

st.set_page_config(
    page_title="IT News Intelligence Dashboard",
    layout="wide"
)


# ---------------------------
# CSS
# ---------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #020617 100%);
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    color: #e5e7eb;
    font-weight: 900;
    letter-spacing: -0.05em;
}

h2, h3 {
    color: #e5e7eb;
    font-weight: 800;
}

[data-testid="stCaptionContainer"] {
    color: #94a3b8;
}

[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(56, 189, 248, 0.28);
    padding: 18px;
    border-radius: 20px;
    box-shadow: 0 0 24px rgba(56, 189, 248, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

[data-testid="stMetricValue"] {
    color: #38bdf8;
    font-weight: 900;
}

.stAlert {
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 16px;
    color: #e0f2fe;
}

.stButton > button {
    background: linear-gradient(135deg, #0284c7, #2563eb);
    color: white;
    border: 0;
    border-radius: 999px;
    padding: 0.5rem 1rem;
    font-weight: 700;
    box-shadow: 0 0 18px rgba(37, 99, 235, 0.25);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #38bdf8, #2563eb);
    color: white;
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.18);
}

.stTextInput input {
    background-color: #020617;
    color: #e5e7eb;
    border: 1px solid #334155;
    border-radius: 12px;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


st.title("IT News Intelligence Dashboard")
st.caption("IT 뉴스 텍스트마이닝 기반 키워드 트렌드 · 언론사 분석 · 유사도 · 네트워크 · 감성/리스크 분석")


# ---------------------------
# AWS / S3
# ---------------------------

AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = st.secrets["AWS_REGION"]
BUCKET_NAME = st.secrets["BUCKET_NAME"]

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

PREFIX = "it_news/IT/processed/"
START_DATE = "20260514"


# ---------------------------
# UI Helpers
# ---------------------------

def section(title, subtitle=None):
    st.markdown("---")
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


def html_card(title, value, desc="", color="#38bdf8"):
    st.markdown(
        f"""
        <div style="
            background: rgba(15,23,42,0.82);
            border: 1px solid rgba(56,189,248,0.25);
            border-radius: 22px;
            padding: 22px;
            min-height: 150px;
            box-shadow: 0 0 24px rgba(56,189,248,0.08);
        ">
            <div style="color:#94a3b8;font-size:14px;font-weight:700;margin-bottom:8px;">
                {title}
            </div>
            <div style="color:{color};font-size:36px;font-weight:900;margin-bottom:8px;">
                {value}
            </div>
            <div style="color:#cbd5e1;font-size:14px;line-height:1.6;">
                {desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_progress_list(df, label_col, value_col, title=None, top_n=10):
    if title:
        st.markdown(f"### {title}")

    if df.empty:
        st.warning("표시할 데이터가 없습니다.")
        return

    view_df = df.head(top_n).copy()
    max_value = view_df[value_col].max()

    colors = [
        "#38bdf8", "#60a5fa", "#818cf8", "#22c55e", "#14b8a6",
        "#eab308", "#f97316", "#ef4444", "#ec4899", "#a855f7"
    ]

    for idx, (_, row) in enumerate(view_df.iterrows()):
        label = row[label_col]
        value = float(row[value_col])
        ratio = value / max_value * 100 if max_value else 0
        color = colors[idx % len(colors)]

        value_text = f"{int(value):,}건" if value == int(value) else f"{value:.2f}"

        st.markdown(
            f"""
            <div style="
                background: rgba(15,23,42,0.75);
                border: 1px solid rgba(148,163,184,0.12);
                border-radius: 18px;
                padding: 18px 22px;
                margin-bottom: 14px;
                box-shadow: 0 0 18px rgba(56,189,248,0.05);
            ">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <div style="font-size:21px;font-weight:900;color:white;">
                        #{idx+1} {label}
                    </div>
                    <div style="font-size:20px;font-weight:900;color:{color};">
                        {value_text}
                    </div>
                </div>
                <div style="width:100%;height:15px;background:#1e293b;border-radius:999px;overflow:hidden;">
                    <div style="
                        width:{ratio}%;
                        height:100%;
                        background: linear-gradient(90deg, {color}, #38bdf8);
                        border-radius:999px;
                        box-shadow:0 0 14px {color};
                    "></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_methodology_cards():
    methods = [
        {
            "title": "TF-IDF",
            "desc": "모든 기사에 흔한 단어보다 특정 기사군에서 상대적으로 중요한 단어에 높은 가중치를 부여합니다."
        },
        {
            "title": "Cosine Similarity",
            "desc": "TF-IDF 벡터 간 각도 유사도를 계산하여 특정 키워드와 유사한 주제의 기사를 찾습니다."
        },
        {
            "title": "Co-occurrence",
            "desc": "같은 기사 안에 함께 등장한 키워드를 분석해 이슈 간 연결 구조를 파악합니다."
        },
        {
            "title": "Sentiment Analysis",
            "desc": "성장·투자·혁신 또는 해킹·유출·침해 키워드로 보도 성향을 분류합니다."
        },
        {
            "title": "Time-Series",
            "desc": "날짜별 기사량 변화를 추적해 특정 이슈가 언제 집중되었는지 확인합니다."
        }
    ]

    cols = st.columns(5)

    for idx, method in enumerate(methods):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="
                    background: rgba(15,23,42,0.82);
                    border: 1px solid rgba(56,189,248,0.25);
                    border-radius: 20px;
                    padding: 20px;
                    min-height: 210px;
                    box-shadow: 0 0 18px rgba(56,189,248,0.08);
                ">
                    <div style="color:#38bdf8;font-size:20px;font-weight:900;margin-bottom:12px;">
                        {method["title"]}
                    </div>
                    <div style="color:#cbd5e1;font-size:14px;line-height:1.7;">
                        {method["desc"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_topic_cards(topic_df):
    cols = st.columns(3)
    colors = ["#38bdf8", "#818cf8", "#22c55e", "#f97316", "#ef4444", "#a855f7"]

    for idx, (_, row) in enumerate(topic_df.iterrows()):
        with cols[idx % 3]:
            html_card(
                row["topic"],
                f"{int(row['count']):,}건",
                row["keywords"],
                colors[idx % len(colors)]
            )


def render_sentiment_cards(sentiment_df):
    cols = st.columns(len(sentiment_df))
    color_map = {
        "긍정/성장": "#22c55e",
        "중립": "#38bdf8",
        "부정/리스크": "#ef4444"
    }

    for idx, (_, row) in enumerate(sentiment_df.iterrows()):
        sentiment = row["sentiment"]
        with cols[idx]:
            html_card(
                sentiment,
                f"{int(row['count']):,}건",
                "기사 제목과 요약문 기반 분류",
                color_map.get(sentiment, "#38bdf8")
            )


# ---------------------------
# Data Helpers
# ---------------------------

def extract_yyyymmdd_from_key(key):
    match = re.search(r"(\d{8})", key)
    return match.group(1) if match else None


def list_all_csv_files(bucket, prefix):
    all_files = []
    token = None

    while True:
        params = {"Bucket": bucket, "Prefix": prefix}
        if token:
            params["ContinuationToken"] = token

        response = s3.list_objects_v2(**params)
        all_files.extend(response.get("Contents", []))

        if response.get("IsTruncated"):
            token = response.get("NextContinuationToken")
        else:
            break

    return [file for file in all_files if file["Key"].endswith(".csv")]


def load_csv_from_s3(key):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    temp_df = pd.read_csv(obj["Body"])
    temp_df["loaded_file"] = key
    return temp_df


def find_date_column(df):
    for col in ["pubDate_dt", "pubDate_ymd", "pubDate", "date", "published_date"]:
        if col in df.columns:
            return col
    return None


def normalize_date_column(df, raw_date_col):
    if not raw_date_col:
        df["analysis_date"] = "unknown"
        return df, "analysis_date"

    df[raw_date_col] = pd.to_datetime(df[raw_date_col], errors="coerce")
    df["analysis_date"] = df[raw_date_col].dt.strftime("%Y-%m-%d")
    df["analysis_date"] = df["analysis_date"].fillna("unknown")
    return df, "analysis_date"


def make_text_series(df):
    text_series = pd.Series("", index=df.index)
    for col in ["title", "description"]:
        if col in df.columns:
            text_series = text_series + " " + df[col].fillna("").astype(str)
    return text_series


def filter_by_keyword(df, keyword):
    text_series = make_text_series(df)
    return df[text_series.str.contains(keyword, case=False, regex=False)]


def filter_by_keywords(df, keywords):
    condition = pd.Series(False, index=df.index)
    text_series = make_text_series(df)

    for keyword in keywords:
        condition = condition | text_series.str.contains(
            keyword,
            case=False,
            regex=False
        )

    return df[condition]


def count_keywords(df, keywords):
    text_series = make_text_series(df)
    rows = []

    for keyword in keywords:
        count = text_series.str.contains(
            keyword,
            case=False,
            regex=False
        ).sum()

        rows.append({
            "keyword": keyword,
            "count": int(count)
        })

    return pd.DataFrame(rows).sort_values("count", ascending=False)


def make_daily_top_keywords(df, keywords, date_col, top_n=5):
    rows = []

    for date in sorted(df[date_col].dropna().unique()):
        date_df = df[df[date_col] == date]
        keyword_df = count_keywords(date_df, keywords).head(top_n)

        for rank, row in enumerate(keyword_df.itertuples(), start=1):
            rows.append({
                "date": date,
                "rank": rank,
                "keyword": row.keyword,
                "count": row.count
            })

    return pd.DataFrame(rows)


def show_article_table(df, date_col="analysis_date"):
    show_cols = [
        col for col in [
            date_col,
            "media_domain",
            "source",
            "title",
            "description",
            "originallink",
            "link"
        ]
        if col in df.columns
    ]

    if date_col in df.columns:
        df = df.sort_values(by=date_col, ascending=False)

    st.dataframe(df[show_cols], use_container_width=True)


# ---------------------------
# Analysis Helpers
# ---------------------------

def make_keyword_network(df, keywords):
    text_series = make_text_series(df)
    rows = []

    for _, text in text_series.items():
        appeared = []

        for keyword in keywords:
            if keyword.lower() in text.lower():
                appeared.append(keyword)

        for a, b in combinations(sorted(set(appeared)), 2):
            rows.append((a, b))

    if not rows:
        return pd.DataFrame(columns=["keyword_a", "keyword_b", "co_count"])

    return (
        pd.DataFrame(rows, columns=["keyword_a", "keyword_b"])
        .value_counts()
        .reset_index(name="co_count")
        .sort_values("co_count", ascending=False)
    )


def build_network_graph(network_df, top_n=25):
    top_df = network_df.head(top_n)
    G = nx.Graph()

    for _, row in top_df.iterrows():
        source = row["keyword_a"]
        target = row["keyword_b"]
        weight = int(row["co_count"])

        G.add_node(source)
        G.add_node(target)
        G.add_edge(source, target, value=weight, title=f"동시출현: {weight}")

    net = Network(height="720px", width="100%", bgcolor="#0f172a", font_color="white")
    net.from_nx(G)

    degree_dict = dict(G.degree())

    for node in net.nodes:
        node_id = node["id"]
        node["size"] = 18 + degree_dict.get(node_id, 1) * 5
        node["color"] = "#38bdf8"
        node["borderWidth"] = 2

    for edge in net.edges:
        edge["color"] = "#64748b"
        edge["smooth"] = True

    net.set_options("""
    var options = {
      "nodes": {"font": {"size": 18, "color": "white"}},
      "edges": {"font": {"size": 12}, "scaling": {"min": 1, "max": 8}},
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -25000,
          "centralGravity": 0.3,
          "springLength": 130,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75
      }
    }
    """)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(temp_file.name)
    return temp_file.name


def classify_sentiment(df):
    positive_words = [
        "성장", "확대", "출시", "투자", "협력", "개선", "강화",
        "수주", "증가", "성공", "최초", "고도화", "혁신"
    ]

    negative_words = [
        "해킹", "침해", "유출", "장애", "중단", "규제", "감소",
        "적자", "위험", "논란", "피해", "취약점", "공격"
    ]

    result = []

    for text in make_text_series(df):
        pos = sum(word in text for word in positive_words)
        neg = sum(word in text for word in negative_words)

        if pos > neg:
            result.append("긍정/성장")
        elif neg > pos:
            result.append("부정/리스크")
        else:
            result.append("중립")

    return result


def make_topic_sentiment_matrix(df, topic_map):
    rows = []

    for topic, keywords in topic_map.items():
        topic_df = filter_by_keywords(df, keywords)
        total = len(topic_df)

        if total == 0:
            rows.append({
                "topic": topic,
                "total": 0,
                "긍정/성장": 0,
                "중립": 0,
                "부정/리스크": 0,
                "positive_ratio": 0,
                "risk_ratio": 0
            })
            continue

        counts = topic_df["sentiment_group"].value_counts()
        positive = int(counts.get("긍정/성장", 0))
        neutral = int(counts.get("중립", 0))
        risk = int(counts.get("부정/리스크", 0))

        rows.append({
            "topic": topic,
            "total": total,
            "긍정/성장": positive,
            "중립": neutral,
            "부정/리스크": risk,
            "positive_ratio": round(positive / total * 100, 1),
            "risk_ratio": round(risk / total * 100, 1)
        })

    return pd.DataFrame(rows).sort_values("total", ascending=False)


def make_topic_timeseries(df, topic_map, date_col):
    rows = []

    for date in sorted(df[date_col].dropna().unique()):
        date_df = df[df[date_col] == date]
        row = {"date": date}

        for topic, keywords in topic_map.items():
            row[topic] = len(filter_by_keywords(date_df, keywords))

        rows.append(row)

    return pd.DataFrame(rows)


def extract_tfidf_keywords(df, top_n=20):
    text_series = make_text_series(df).fillna("").astype(str)

    if len(text_series) == 0 or text_series.str.strip().eq("").all():
        return pd.DataFrame(columns=["keyword", "score"])

    vectorizer = TfidfVectorizer(
        max_features=1000,
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9]{2,}\b"
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(text_series)
    except ValueError:
        return pd.DataFrame(columns=["keyword", "score"])

    scores = tfidf_matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame({"keyword": words, "score": scores})
    return tfidf_df.sort_values("score", ascending=False).head(top_n)


def make_cosine_similarity_articles(df, keyword, top_n=10):
    text_series = make_text_series(df).fillna("").astype(str)

    if len(text_series) < 2:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(
        max_features=1200,
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9]{2,}\b"
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(text_series.tolist() + [keyword])
        article_matrix = tfidf_matrix[:-1]
        query_vector = tfidf_matrix[-1]
        scores = cosine_similarity(article_matrix, query_vector).flatten()
    except ValueError:
        return pd.DataFrame()

    result_df = df.copy()
    result_df["similarity_score"] = scores

    result_cols = [
        col for col in [
            "analysis_date",
            "media_domain",
            "source",
            "title",
            "description",
            "originallink",
            "link",
            "similarity_score"
        ]
        if col in result_df.columns
    ]

    return result_df.sort_values("similarity_score", ascending=False)[result_cols].head(top_n)


def make_media_frame_analysis(df, media_col="media_domain"):
    frame_keywords = {
        "성장/혁신": ["성장", "혁신", "출시", "확대", "협력", "투자", "강화"],
        "보안/리스크": ["해킹", "침해", "유출", "장애", "공격", "위험", "취약점"],
        "산업/경쟁": ["시장", "경쟁", "점유율", "HBM", "반도체", "공급망", "수출"],
        "정책/규제": ["정부", "규제", "정책", "법안", "지원", "제도"],
        "인프라/클라우드": ["클라우드", "데이터센터", "AWS", "Azure", "서버", "인프라"]
    }

    rows = []
    top_media = df[media_col].fillna("unknown").value_counts().head(15).index.tolist()

    for media in top_media:
        media_df = df[df[media_col] == media]
        row = {
            "media_domain": media,
            "total_articles": len(media_df)
        }

        text_series = make_text_series(media_df)

        for frame, keywords in frame_keywords.items():
            count = 0
            for keyword in keywords:
                count += text_series.str.contains(keyword, case=False, regex=False).sum()
            row[frame] = int(count)

        rows.append(row)

    return pd.DataFrame(rows)


def make_media_keyword_analysis(df, keywords, media_col="media_domain"):
    rows = []
    top_media = df[media_col].fillna("unknown").value_counts().head(15).index.tolist()

    for media in top_media:
        media_df = df[df[media_col] == media]
        row = {
            "media_domain": media,
            "total_articles": len(media_df)
        }

        keyword_df = count_keywords(media_df, keywords)

        for _, item in keyword_df.iterrows():
            row[item["keyword"]] = int(item["count"])

        rows.append(row)

    return pd.DataFrame(rows)


def make_media_sentiment_analysis(df, media_col="media_domain"):
    rows = []
    top_media = df[media_col].fillna("unknown").value_counts().head(15).index.tolist()

    for media in top_media:
        media_df = df[df[media_col] == media]
        total = len(media_df)
        counts = media_df["sentiment_group"].value_counts()

        positive = int(counts.get("긍정/성장", 0))
        neutral = int(counts.get("중립", 0))
        risk = int(counts.get("부정/리스크", 0))

        rows.append({
            "media_domain": media,
            "total_articles": total,
            "긍정/성장": positive,
            "중립": neutral,
            "부정/리스크": risk,
            "positive_ratio": round(positive / total * 100, 1) if total else 0,
            "risk_ratio": round(risk / total * 100, 1) if total else 0
        })

    return pd.DataFrame(rows)


def make_event_annotations(df, date_col):
    rows = []
    date_counts = df[date_col].value_counts().sort_values(ascending=False).head(5)

    for date, count in date_counts.items():
        rows.append({
            "date": date,
            "event": f"기사 급증 ({count:,}건)",
            "analysis": "주요 기업 발표, 보안 사고, 기술 행사, 정책 발표 등 실제 이벤트와 연결 가능"
        })

    return pd.DataFrame(rows)


# ---------------------------
# Load Data
# ---------------------------

csv_files = list_all_csv_files(BUCKET_NAME, PREFIX)

filtered_files = []

for file in csv_files:
    file_date = extract_yyyymmdd_from_key(file["Key"])
    if file_date and file_date >= START_DATE:
        filtered_files.append(file)

if not filtered_files:
    st.error("2026년 5월 14일 이후 CSV 파일을 찾지 못했습니다.")
    st.stop()

filtered_files = sorted(filtered_files, key=lambda x: x["Key"])
df_list = [load_csv_from_s3(file["Key"]) for file in filtered_files]

df = pd.concat(df_list, ignore_index=True)

raw_date_col = find_date_column(df)
df, date_col = normalize_date_column(df, raw_date_col)

dedup_cols = [col for col in ["originallink", "link", "title"] if col in df.columns]

if dedup_cols:
    df = df.drop_duplicates(subset=dedup_cols, keep="last")

if "source" not in df.columns:
    df["source"] = "unknown"

if "media_domain" not in df.columns:
    df["media_domain"] = df["source"]

df["media_domain"] = df["media_domain"].fillna("unknown")
df["source"] = df["source"].fillna("unknown")

latest_date = df[date_col].dropna().max()
latest_df = df[df[date_col] == latest_date].copy()


# ---------------------------
# Analysis Config
# ---------------------------

main_keywords = [
    "AI", "인공지능", "생성형AI", "챗GPT",
    "반도체", "클라우드", "보안", "데이터",
    "로봇", "배터리", "전기차",
    "삼성", "네이버", "카카오", "엔비디아",
    "AWS", "Azure", "해킹", "개인정보", "데이터센터", "HBM"
]

topic_map = {
    "AI/인공지능": ["AI", "인공지능", "생성형AI", "챗GPT", "엔비디아"],
    "반도체": ["반도체", "삼성전자", "SK하이닉스", "칩", "HBM"],
    "클라우드/데이터센터": ["클라우드", "AWS", "Azure", "데이터센터"],
    "보안/개인정보": ["보안", "해킹", "개인정보", "랜섬웨어", "침해"],
    "모빌리티/로봇": ["전기차", "자율주행", "배터리", "로봇"],
    "플랫폼/빅테크": ["네이버", "카카오", "구글", "애플", "메타"]
}

df["sentiment_group"] = classify_sentiment(df)
latest_df["sentiment_group"] = classify_sentiment(latest_df)

latest_keyword_df = count_keywords(latest_df, main_keywords)
top_keywords = latest_keyword_df.head(10)

daily_keyword_df = make_daily_top_keywords(df, main_keywords, date_col, top_n=5)
network_df = make_keyword_network(df, main_keywords)
topic_sentiment_df = make_topic_sentiment_matrix(df, topic_map)
topic_timeseries_df = make_topic_timeseries(df, topic_map, date_col)


# ---------------------------
# Dashboard
# ---------------------------

st.subheader("오늘의 IT 뉴스 요약")

c1, c2, c3, c4 = st.columns(4)

with c1:
    html_card("전체 기사 수", f"{len(df):,}", "중복 제거 후 분석 대상 기사")

with c2:
    html_card(f"{latest_date} 기사 수", f"{len(latest_df):,}", "최신일 기준 수집 기사")

with c3:
    html_card("실제 언론사 수", f"{df['media_domain'].nunique():,}", "media_domain 기준")

with c4:
    html_card("수집 파일 수", f"{len(filtered_files):,}", "S3 processed CSV 파일")


section("분석 방법론", "뉴스 텍스트마이닝 기반의 5가지 분석 방법입니다.")
render_methodology_cards()

st.info(
    "TF-IDF는 흔한 단어보다 특정 기사군에서 중요한 단어를 찾고, Cosine Similarity는 TF-IDF 벡터 간 유사도를 계산합니다. "
    "Co-occurrence는 키워드 간 연결 구조를, Sentiment는 보도 성향을, Time-Series는 이슈의 시간적 변화를 분석합니다."
)


section("1. 오늘의 주요 IT 키워드 트렌드")
render_progress_list(top_keywords, "keyword", "count", "오늘의 주요 IT 키워드 TOP 10")


section("2. 실제 언론사 기준 분석", "source는 수집 경로, media_domain은 실제 기사 발행 언론사입니다.")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 수집 경로별 기사 수")
    source_count = df["source"].fillna("unknown").value_counts().reset_index()
    source_count.columns = ["source", "count"]
    st.dataframe(source_count, use_container_width=True)

with col_b:
    st.markdown("### 실제 언론사별 기사 수")
    media_count = df["media_domain"].fillna("unknown").value_counts().reset_index()
    media_count.columns = ["media_domain", "count"]
    st.dataframe(media_count.head(20), use_container_width=True)

st.markdown("### 언론사별 주요 키워드")

media_keyword_df = make_media_keyword_analysis(df, main_keywords)
st.dataframe(media_keyword_df, use_container_width=True)

if not media_keyword_df.empty:
    selected_media = st.selectbox(
        "언론사 선택",
        media_keyword_df["media_domain"].tolist(),
        key="media_keyword_select"
    )

    selected_media_df = df[df["media_domain"] == selected_media]
    selected_media_keywords = count_keywords(selected_media_df, main_keywords).head(10)

    render_progress_list(
        selected_media_keywords,
        "keyword",
        "count",
        f"{selected_media} 주요 키워드 TOP 10"
    )

    with st.expander(f"{selected_media} 기사 보기"):
        show_article_table(selected_media_df, date_col)


section("3. TF-IDF 기반 핵심 키워드 분석", "단순 빈도가 아니라 특정 기사군에서 상대적으로 중요한 키워드를 추출합니다.")

tfidf_df = extract_tfidf_keywords(latest_df, top_n=20)
st.dataframe(tfidf_df, use_container_width=True)

if not tfidf_df.empty:
    tfidf_view = tfidf_df.copy()
    tfidf_view["score_view"] = (tfidf_view["score"] * 100).round(2)

    render_progress_list(
        tfidf_view.head(10),
        "keyword",
        "score_view",
        "TF-IDF 중요 키워드 TOP 10"
    )


section("4. Cosine Similarity 기반 유사 기사 분석", "TF-IDF 벡터 간 각도 유사도를 계산해 특정 키워드와 가까운 기사를 찾습니다.")

similarity_keyword = st.selectbox(
    "유사 기사 분석 키워드 선택",
    main_keywords,
    key="similarity_keyword"
)

similarity_df = make_cosine_similarity_articles(
    df,
    similarity_keyword,
    top_n=10
)

if similarity_df.empty:
    st.warning("선택한 키워드에 대한 유사 기사 분석 결과가 없습니다.")
else:
    st.dataframe(similarity_df, use_container_width=True)

st.info(
    "코사인 유사도는 1에 가까울수록 유사한 주제의 기사입니다. "
    "예를 들어 AI와 반도체·HBM·클라우드 관련 기사가 함께 높게 나타나면 AI 인프라 생태계 이슈로 해석할 수 있습니다."
)


section("5. 일별 주요 IT 키워드", "날짜별 TOP 키워드 변화로 이슈 흐름을 확인합니다.")

st.dataframe(daily_keyword_df, use_container_width=True)

selected_daily_keyword = st.selectbox(
    "키워드별 일자 추이 확인",
    main_keywords
)

selected_daily_keyword_df = daily_keyword_df[
    daily_keyword_df["keyword"] == selected_daily_keyword
]

st.dataframe(selected_daily_keyword_df, use_container_width=True)


section("6. 키워드 클릭해서 관련 기사 보기")

if "selected_keyword" not in st.session_state:
    st.session_state["selected_keyword"] = (
        top_keywords.iloc[0]["keyword"]
        if not top_keywords.empty
        else ""
    )

button_cols = st.columns(5)

for idx, row in top_keywords.reset_index(drop=True).iterrows():
    keyword = row["keyword"]
    count = row["count"]

    with button_cols[idx % 5]:
        if st.button(f"{keyword} ({count})"):
            st.session_state["selected_keyword"] = keyword

selected_keyword = st.session_state["selected_keyword"]

if selected_keyword:
    keyword_articles = filter_by_keyword(df, selected_keyword)

    st.info(
        f"선택된 키워드: {selected_keyword} / 관련 기사 수: {len(keyword_articles):,}건"
    )

    show_article_table(keyword_articles, date_col)


section(f"7. {latest_date} 주요 이슈 자동 요약", "키워드를 주제군으로 묶어 최신일 이슈 구조를 보여줍니다.")

issue_rows = []

for topic, keywords in topic_map.items():
    topic_df = filter_by_keywords(latest_df, keywords)

    issue_rows.append({
        "topic": topic,
        "count": len(topic_df),
        "keywords": ", ".join(keywords)
    })

issue_df = pd.DataFrame(issue_rows).sort_values("count", ascending=False)

render_topic_cards(issue_df)
st.dataframe(issue_df, use_container_width=True)


section("8. 키워드 동시출현 네트워크 분석", "같은 기사 안에 함께 등장한 키워드 조합을 계산합니다.")

st.dataframe(network_df.head(30), use_container_width=True)


section("9. 키워드 네트워크 그래프 시각화", "동시출현 관계를 네트워크 그래픽으로 표현합니다.")

if not network_df.empty:
    html_path = build_network_graph(network_df, top_n=25)

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    components.html(
        html_content,
        height=760,
        scrolling=True
    )
else:
    st.warning("네트워크 그래프를 생성할 수 있는 데이터가 없습니다.")


section("10. 주제별 감성 지수 교차 분석", "어떤 기술 주제가 긍정적으로, 어떤 주제가 리스크 중심으로 보도되는지 비교합니다.")

st.dataframe(topic_sentiment_df, use_container_width=True)

sentiment_highlight = topic_sentiment_df.sort_values(
    "risk_ratio",
    ascending=False
).head(3)

cols = st.columns(3)

for idx, (_, row) in enumerate(sentiment_highlight.iterrows()):
    with cols[idx]:
        html_card(
            row["topic"],
            f"{row['risk_ratio']}%",
            "부정/리스크 보도 비율",
            "#ef4444"
        )


section("11. 주제별 시계열 트렌드", "날짜별 주제 기사 수 변화를 통해 이슈의 생애주기를 봅니다.")

st.dataframe(topic_timeseries_df, use_container_width=True)

selected_timeseries_topic = st.selectbox(
    "시계열 상세 확인 주제",
    list(topic_map.keys()),
    key="timeseries_topic"
)

selected_topic_daily_df = topic_timeseries_df[
    ["date", selected_timeseries_topic]
].sort_values(selected_timeseries_topic, ascending=False)

st.dataframe(selected_topic_daily_df, use_container_width=True)

if not selected_topic_daily_df.empty:
    top_topic_date = selected_topic_daily_df.iloc[0]["date"]

    st.info(
        f"{selected_timeseries_topic} 관련 기사가 가장 많았던 날짜는 {top_topic_date}입니다. "
        "실제 사건과 연결해 해석하면 좋습니다."
    )


section("12. 언론사별 보도 주제 비중", "각 언론사가 어떤 IT 주제를 많이 다루는지 비교합니다.")

media_topic_rows = []
top_media = df["media_domain"].value_counts().head(15).index.tolist()

for media in top_media:
    media_df = df[df["media_domain"] == media]
    total = len(media_df)

    row = {
        "media_domain": media,
        "total": total
    }

    for topic, keywords in topic_map.items():
        count = len(filter_by_keywords(media_df, keywords))
        row[topic] = round(count / total * 100, 1) if total else 0

    media_topic_rows.append(row)

media_topic_df = pd.DataFrame(media_topic_rows)
st.dataframe(media_topic_df, use_container_width=True)


section("13. 언론사별 보도 프레임 분석", "성장/혁신, 보안/리스크, 산업/경쟁 등 보도 관점 차이를 비교합니다.")

frame_df = make_media_frame_analysis(df)
st.dataframe(frame_df, use_container_width=True)

selected_frame = st.selectbox(
    "프레임 선택",
    ["성장/혁신", "보안/리스크", "산업/경쟁", "정책/규제", "인프라/클라우드"]
)

if selected_frame in frame_df.columns:
    frame_view = frame_df[["media_domain", selected_frame]].sort_values(
        selected_frame,
        ascending=False
    ).head(10)

    render_progress_list(
        frame_view,
        "media_domain",
        selected_frame,
        f"{selected_frame} 프레임 TOP 10"
    )


section("보조 분석: 감성/리스크 전체 분포")

sentiment_count = df["sentiment_group"].value_counts().reset_index()
sentiment_count.columns = ["sentiment", "count"]

render_sentiment_cards(sentiment_count)

with st.expander("부정/리스크 기사 보기"):
    show_article_table(
        df[df["sentiment_group"] == "부정/리스크"],
        date_col
    )


section("보조 분석: 이벤트 주석 기반 시계열")

event_df = make_event_annotations(df, date_col)
st.dataframe(event_df, use_container_width=True)


section("전체 기사 검색")

search_text = st.text_input("검색어를 입력하세요")

if search_text:
    search_df = filter_by_keyword(df, search_text)
    st.write(f"검색 결과: {len(search_df):,}건")
    show_article_table(search_df, date_col)


with st.expander("원본 데이터 보기"):
    st.dataframe(df, use_container_width=True)

with st.expander("불러온 파일 목록"):
    for file in filtered_files:
        st.write(file["Key"])
