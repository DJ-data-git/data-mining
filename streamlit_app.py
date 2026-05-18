import streamlit as st
import pandas as pd
import boto3
import re
from collections import Counter
from itertools import combinations

# ---------------------------
# 페이지 설정
# ---------------------------

st.set_page_config(
    page_title="IT News Intelligence Dashboard",
    layout="wide"
)

# ---------------------------
# 디자인 CSS
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
    color: #38bdf8;
    font-weight: 800;
    letter-spacing: -0.04em;
}

h2, h3 {
    color: #e5e7eb;
    font-weight: 700;
}

[data-testid="stCaptionContainer"] {
    color: #94a3b8;
}

[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 0 24px rgba(56, 189, 248, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

[data-testid="stMetricValue"] {
    color: #38bdf8;
    font-weight: 800;
}

.stAlert {
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 14px;
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
    border: 0;
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.2);
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
st.caption("실시간 IT 뉴스 기반 텍스트마이닝 · 키워드 트렌드 · 네트워크 · 감성/리스크 분석")

# ---------------------------
# AWS / S3 설정
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
# 함수
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

    condition = text_series.str.contains(
        keyword,
        case=False,
        regex=False
    )

    return df[condition]


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
    result = []

    for keyword in keywords:
        count = text_series.str.contains(
            keyword,
            case=False,
            regex=False
        ).sum()

        result.append({
            "keyword": keyword,
            "count": int(count)
        })

    return pd.DataFrame(result).sort_values("count", ascending=False)


def extract_simple_words(df, top_n=20):
    text = " ".join(make_text_series(df).tolist())
    words = re.findall(r"[가-힣A-Za-z0-9]{2,}", text)

    stopwords = {
        "기자", "뉴스", "오늘", "이번", "대한", "관련", "통해", "위해",
        "있는", "했다", "한다", "지난", "오는", "올해", "기업", "산업",
        "서비스", "시장", "기술", "제공", "발표", "추진", "구축", "사용",
        "것으로", "이라고", "에서", "으로", "하고", "등을", "등의",
        "위한", "밝혔다", "naver", "google"
    }

    words = [word for word in words if word.lower() not in stopwords]
    counter = Counter(words)

    return pd.DataFrame(counter.most_common(top_n), columns=["keyword", "count"])


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


def classify_sentiment(df):
    positive_words = [
        "성장", "확대", "출시", "투자", "협력", "개선", "강화",
        "수주", "증가", "성공", "최초", "고도화", "혁신"
    ]

    negative_words = [
        "해킹", "침해", "유출", "장애", "중단", "규제", "감소",
        "적자", "위험", "논란", "피해", "취약점", "공격"
    ]

    text_series = make_text_series(df)
    result = []

    for text in text_series:
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


def make_auto_insight(topic_sentiment_df, network_df):
    insights = []

    if not topic_sentiment_df.empty:
        top_positive = topic_sentiment_df.sort_values("positive_ratio", ascending=False).iloc[0]
        top_risk = topic_sentiment_df.sort_values("risk_ratio", ascending=False).iloc[0]

        insights.append(
            f"긍정 보도 비중이 가장 높은 주제는 '{top_positive['topic']}'이며, "
            f"긍정/성장 기사 비율은 {top_positive['positive_ratio']}%입니다."
        )

        insights.append(
            f"부정/리스크 보도 비중이 가장 높은 주제는 '{top_risk['topic']}'이며, "
            f"부정/리스크 기사 비율은 {top_risk['risk_ratio']}%입니다."
        )

    if not network_df.empty:
        top_pair = network_df.iloc[0]

        insights.append(
            f"가장 강하게 함께 등장한 키워드 조합은 "
            f"'{top_pair['keyword_a']} - {top_pair['keyword_b']}'이며, "
            f"동시출현 빈도는 {top_pair['co_count']}건입니다."
        )

    return insights


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
# 데이터 로딩
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

if "media_domain" not in df.columns:
    if "source" in df.columns:
        df["media_domain"] = df["source"]
    else:
        df["media_domain"] = "unknown"

latest_date = df[date_col].dropna().max()
latest_df = df[df[date_col] == latest_date].copy()

# ---------------------------
# 분석 기준
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

daily_keyword_df = make_daily_top_keywords(
    df,
    main_keywords,
    date_col,
    top_n=5
)

network_df = make_keyword_network(df, main_keywords)
topic_sentiment_df = make_topic_sentiment_matrix(df, topic_map)
topic_timeseries_df = make_topic_timeseries(df, topic_map, date_col)

# ---------------------------
# 0. 최상단 트렌드 요약
# ---------------------------

st.subheader("오늘의 주요 IT 키워드 트렌드")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("전체 기사 수", len(df))

with col2:
    st.metric(f"{latest_date} 기사 수", len(latest_df))

with col3:
    st.metric("언론사 수", df["media_domain"].nunique())

with col4:
    st.metric("수집 파일 수", len(filtered_files))

st.markdown("### 오늘의 주요 IT 키워드 TOP 10")
st.bar_chart(top_keywords.set_index("keyword"))

st.dataframe(top_keywords, use_container_width=True)

st.markdown("### 오늘의 메가 트렌드 해석")

insights = make_auto_insight(topic_sentiment_df, network_df)

for insight in insights:
    st.info(insight)

st.info(
    "해석 포인트: AI가 단독 이슈로만 등장하는 것이 아니라 반도체, HBM, 클라우드, 데이터센터와 함께 등장한다면 "
    "최근 IT 뉴스의 흐름은 AI 모델 자체를 넘어 연산 인프라와 하드웨어 생태계로 확장되고 있다고 볼 수 있습니다."
)

st.markdown("---")

# ---------------------------
# 1. 일별 주요 IT 키워드
# ---------------------------

st.subheader("일별 주요 IT 키워드")

st.caption("각 날짜별로 기사 제목과 설명에서 많이 등장한 주요 IT 키워드 TOP 5입니다.")

st.dataframe(daily_keyword_df, use_container_width=True)

if not daily_keyword_df.empty:
    pivot_daily_keyword = daily_keyword_df.pivot_table(
        index="date",
        columns="keyword",
        values="count",
        aggfunc="sum",
        fill_value=0
    )

    st.line_chart(pivot_daily_keyword)

selected_daily_keyword = st.selectbox(
    "키워드별 일자 추이 확인",
    main_keywords
)

selected_daily_keyword_df = daily_keyword_df[
    daily_keyword_df["keyword"] == selected_daily_keyword
]

st.write(f"'{selected_daily_keyword}' 일별 등장 추이")
st.dataframe(selected_daily_keyword_df, use_container_width=True)

if not selected_daily_keyword_df.empty:
    st.line_chart(
        selected_daily_keyword_df.set_index("date")["count"]
    )

st.markdown("---")

# ---------------------------
# 2. 키워드 클릭 → 관련 기사
# ---------------------------

st.subheader("키워드 클릭해서 관련 기사 보기")

if "selected_keyword" not in st.session_state:
    st.session_state["selected_keyword"] = top_keywords.iloc[0]["keyword"] if not top_keywords.empty else ""

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

    st.info(f"선택된 키워드: {selected_keyword} / 관련 기사 수: {len(keyword_articles)}건")
    show_article_table(keyword_articles, date_col)

st.markdown("---")

# ---------------------------
# 3. 최신일 주요 이슈 자동 요약
# ---------------------------

st.subheader(f"{latest_date} 주요 이슈 자동 요약")

issue_rows = []

for topic, keywords in topic_map.items():
    topic_df = filter_by_keywords(latest_df, keywords)

    issue_rows.append({
        "issue": topic,
        "article_count": len(topic_df),
        "keywords": ", ".join(keywords)
    })

issue_df = pd.DataFrame(issue_rows).sort_values("article_count", ascending=False)

st.dataframe(issue_df, use_container_width=True)
st.bar_chart(issue_df.set_index("issue")["article_count"])

st.markdown("---")

# ---------------------------
# 4. 키워드 동시출현 네트워크 분석
# ---------------------------

st.subheader("키워드 동시출현 네트워크 분석")

st.caption("같은 기사 안에 함께 등장한 키워드 조합을 계산한 결과입니다.")

st.dataframe(network_df.head(30), use_container_width=True)

if not network_df.empty:
    top_network = network_df.head(15).copy()
    top_network["pair"] = top_network["keyword_a"] + " - " + top_network["keyword_b"]

    st.bar_chart(top_network.set_index("pair")["co_count"])

st.markdown("---")

# ---------------------------
# 5. 주제별 기사 수 및 상세 기사
# ---------------------------

st.subheader("주제별 기사 수")

topic_result = []

for topic, keywords in topic_map.items():
    topic_df = filter_by_keywords(df, keywords)

    topic_result.append({
        "topic": topic,
        "count": len(topic_df),
        "keywords": ", ".join(keywords)
    })

topic_df = pd.DataFrame(topic_result).sort_values("count", ascending=False)

st.dataframe(topic_df, use_container_width=True)
st.bar_chart(topic_df.set_index("topic")["count"])

selected_topic = st.selectbox(
    "주제별 기사 상세보기",
    topic_df["topic"].tolist()
)

selected_topic_df = filter_by_keywords(df, topic_map[selected_topic])

st.write(f"{selected_topic} 관련 기사 수: {len(selected_topic_df)}건")
show_article_table(selected_topic_df, date_col)

st.markdown("---")

# ---------------------------
# 6. 주제별 감성 지수 교차 분석
# ---------------------------

st.subheader("주제별 감성 지수 교차 분석")

st.caption(
    "각 IT 주제별로 긍정/성장, 중립, 부정/리스크 보도가 어떻게 분포하는지 분석합니다."
)

st.dataframe(topic_sentiment_df, use_container_width=True)

st.markdown("### 주제별 긍정 보도 비율")
st.bar_chart(topic_sentiment_df.set_index("topic")["positive_ratio"])

st.markdown("### 주제별 부정/리스크 보도 비율")
st.bar_chart(topic_sentiment_df.set_index("topic")["risk_ratio"])

st.info(
    "해석 예시: AI·클라우드 등 신기술 주제는 성장, 투자, 혁신 중심의 긍정 보도가 많고, "
    "보안·개인정보 주제는 해킹, 유출, 침해 등 리스크 관점의 보도가 상대적으로 강하게 나타날 수 있습니다."
)

st.markdown("---")

# ---------------------------
# 7. 주제별 시계열 트렌드
# ---------------------------

st.subheader("주제별 시계열 트렌드 분석")

st.caption(
    "날짜별로 각 주제의 기사 수 변화를 비교하여 특정 이슈가 어느 시점에 집중되었는지 확인합니다."
)

st.dataframe(topic_timeseries_df, use_container_width=True)

if not topic_timeseries_df.empty:
    st.line_chart(topic_timeseries_df.set_index("date"))

selected_timeseries_topic = st.selectbox(
    "시계열 상세 확인 주제",
    list(topic_map.keys()),
    key="timeseries_topic"
)

selected_topic_daily_df = topic_timeseries_df[
    ["date", selected_timeseries_topic]
].sort_values(selected_timeseries_topic, ascending=False)

st.write(f"{selected_timeseries_topic} 기사 수가 많았던 날짜")
st.dataframe(selected_topic_daily_df, use_container_width=True)

if not selected_topic_daily_df.empty:
    top_topic_date = selected_topic_daily_df.iloc[0]["date"]

    st.info(
        f"{selected_timeseries_topic} 관련 기사가 가장 많았던 날짜는 {top_topic_date}입니다. "
        "보고서에서는 해당 날짜의 실제 주요 사건이나 기업 발표와 연결해 해석하면 좋습니다."
    )

st.markdown("---")

# ---------------------------
# 8. 언론사별 보도 주제 비중
# ---------------------------

st.subheader("언론사별 보도 주제 비중")

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
        ratio = round(count / total * 100, 1) if total else 0
        row[topic] = ratio

    media_topic_rows.append(row)

media_topic_df = pd.DataFrame(media_topic_rows)

st.caption("각 언론사의 전체 기사 중 주제별 기사 비중입니다. 단위: %")
st.dataframe(media_topic_df, use_container_width=True)

st.markdown("---")

# ---------------------------
# 9. 감성/리스크 분석
# ---------------------------

st.subheader("간단 감성/리스크 분석")

sentiment_count = df["sentiment_group"].value_counts().reset_index()
sentiment_count.columns = ["sentiment", "count"]

st.dataframe(sentiment_count, use_container_width=True)
st.bar_chart(sentiment_count.set_index("sentiment"))

risk_df = df[df["sentiment_group"] == "부정/리스크"]

with st.expander("부정/리스크 기사 보기"):
    show_article_table(risk_df, date_col)

st.markdown("---")

# ---------------------------
# 10. 날짜별 기사 수
# ---------------------------

st.subheader("날짜별 기사 수")

date_count = df[date_col].value_counts().sort_index().reset_index()
date_count.columns = ["date", "count"]

st.dataframe(date_count, use_container_width=True)
st.line_chart(date_count.set_index("date"))

st.markdown("---")

# ---------------------------
# 11. 언론사별 기사 수
# ---------------------------

st.subheader("언론사별 기사 수")

media_count = df["media_domain"].fillna("unknown").value_counts().reset_index()
media_count.columns = ["media_domain", "count"]

st.dataframe(media_count, use_container_width=True)
st.bar_chart(media_count.set_index("media_domain"))

st.markdown("---")

# ---------------------------
# 12. 전체 기사 검색
# ---------------------------

st.subheader("전체 기사 검색")

search_text = st.text_input("검색어를 입력하세요")

if search_text:
    search_df = filter_by_keyword(df, search_text)

    st.write(f"검색 결과: {len(search_df)}건")
    show_article_table(search_df, date_col)

st.markdown("---")

# ---------------------------
# 원본 데이터
# ---------------------------

with st.expander("원본 데이터 보기"):
    st.dataframe(df, use_container_width=True)

with st.expander("불러온 파일 목록"):
    for file in filtered_files:
        st.write(file["Key"])
