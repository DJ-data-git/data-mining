import streamlit as st
import pandas as pd
import boto3
import re
from collections import Counter

# ---------------------------
# 페이지 설정
# ---------------------------

st.set_page_config(
    page_title="IT 뉴스 분석 대시보드",
    layout="wide"
)

st.title("IT 뉴스 분석 대시보드")

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

# ---------------------------
# S3 경로 설정
# ---------------------------

PREFIX = "it_news/IT/processed/"
START_DATE = "20260514"


# ---------------------------
# 함수
# ---------------------------

def extract_yyyymmdd_from_key(key):
    match = re.search(r"(\d{8})", key)
    if match:
        return match.group(1)
    return None


def list_all_csv_files(bucket, prefix):
    all_files = []
    token = None

    while True:
        params = {
            "Bucket": bucket,
            "Prefix": prefix
        }

        if token:
            params["ContinuationToken"] = token

        response = s3.list_objects_v2(**params)
        all_files.extend(response.get("Contents", []))

        if response.get("IsTruncated"):
            token = response.get("NextContinuationToken")
        else:
            break

    return [
        file for file in all_files
        if file["Key"].endswith(".csv")
    ]


def load_csv_from_s3(key):
    obj = s3.get_object(
        Bucket=BUCKET_NAME,
        Key=key
    )

    temp_df = pd.read_csv(obj["Body"])
    temp_df["loaded_file"] = key

    return temp_df


def find_date_column(df):
    possible_date_cols = [
        "pubDate_dt",
        "pubDate_ymd",
        "pubDate",
        "date",
        "published_date"
    ]

    for col in possible_date_cols:
        if col in df.columns:
            return col

    return None


def normalize_date_column(df, date_col):
    if not date_col:
        return df, None

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df["analysis_date"] = df[date_col].dt.strftime("%Y-%m-%d")

    return df, "analysis_date"


def make_text_series(df):
    text_series = pd.Series("", index=df.index)

    for col in ["title", "description"]:
        if col in df.columns:
            text_series = text_series + " " + df[col].fillna("").astype(str)

    return text_series


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

    return pd.DataFrame(result).sort_values(
        "count",
        ascending=False
    )


def filter_by_keyword(df, keyword):
    condition = pd.Series(False, index=df.index)

    for col in ["title", "description"]:
        if col in df.columns:
            condition = condition | df[col].fillna("").astype(str).str.contains(
                keyword,
                case=False,
                regex=False
            )

    return df[condition]


def extract_simple_words(df, top_n=20):
    text = " ".join(make_text_series(df).tolist())

    words = re.findall(r"[가-힣A-Za-z0-9]{2,}", text)

    stopwords = {
        "기자", "뉴스", "오늘", "이번", "대한", "관련", "통해", "위해",
        "있는", "했다", "한다", "지난", "오는", "올해", "기업", "산업",
        "서비스", "시장", "기술", "제공", "발표", "추진", "구축", "사용",
        "naver", "google", "것으로", "이라고", "에서", "으로", "하고",
        "등을", "등의", "위한", "밝혔다"
    }

    words = [
        word for word in words
        if word.lower() not in stopwords
    ]

    counter = Counter(words)

    return pd.DataFrame(
        counter.most_common(top_n),
        columns=["keyword", "count"]
    )


def filter_by_topic(df, keywords):
    condition = pd.Series(False, index=df.index)

    for keyword in keywords:
        condition = condition | filter_by_keyword(df, keyword).index.to_series().isin(df.index)

    return df[condition]


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

filtered_files = sorted(
    filtered_files,
    key=lambda x: x["Key"]
)

df_list = []

for file in filtered_files:
    df_list.append(load_csv_from_s3(file["Key"]))

df = pd.concat(df_list, ignore_index=True)

# ---------------------------
# 날짜 컬럼 처리
# ---------------------------

raw_date_col = find_date_column(df)
df, date_col = normalize_date_column(df, raw_date_col)

if date_col:
    latest_date = df[date_col].dropna().max()
    latest_df = df[df[date_col] == latest_date].copy()
else:
    latest_date = "unknown"
    latest_df = df.copy()
    st.warning("날짜 컬럼을 찾지 못했습니다.")

# ---------------------------
# 중복 제거
# ---------------------------

dedup_cols = []

for col in ["originallink", "link", "title"]:
    if col in df.columns:
        dedup_cols.append(col)

if dedup_cols:
    df = df.drop_duplicates(subset=dedup_cols, keep="last")

if date_col:
    latest_df = df[df[date_col] == latest_date].copy()
else:
    latest_df = df.copy()

# ---------------------------
# 분석 키워드
# ---------------------------

main_keywords = [
    "AI",
    "인공지능",
    "생성형AI",
    "챗GPT",
    "반도체",
    "클라우드",
    "보안",
    "데이터",
    "로봇",
    "배터리",
    "전기차",
    "삼성",
    "네이버",
    "카카오",
    "엔비디아"
]

# ---------------------------
# 상단 요약
# ---------------------------

st.subheader(f"{latest_date} 주요 IT 뉴스 요약")

latest_keyword_df = count_keywords(latest_df, main_keywords)
top_keywords = latest_keyword_df.head(10)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("전체 기사 수", len(df))

with col2:
    st.metric(f"{latest_date} 기사 수", len(latest_df))

with col3:
    if "media_domain" in df.columns:
        st.metric("언론사 수", df["media_domain"].nunique())
    elif "source" in df.columns:
        st.metric("수집 소스 수", df["source"].nunique())
    else:
        st.metric("언론사 수", "확인 불가")

with col4:
    st.metric("수집 파일 수", len(filtered_files))

st.write("최신일 주요 키워드 TOP 10")
st.bar_chart(top_keywords.set_index("keyword"))

# ---------------------------
# 키워드 클릭형 필터
# ---------------------------

st.subheader("키워드 클릭해서 기사 보기")

if "selected_keyword" not in st.session_state:
    if not top_keywords.empty:
        st.session_state["selected_keyword"] = top_keywords.iloc[0]["keyword"]
    else:
        st.session_state["selected_keyword"] = ""

button_cols = st.columns(5)

for idx, row in top_keywords.reset_index(drop=True).iterrows():
    keyword = row["keyword"]
    count = row["count"]

    with button_cols[idx % 5]:
        if st.button(f"{keyword} ({count})"):
            st.session_state["selected_keyword"] = keyword

selected_keyword = st.session_state["selected_keyword"]

if selected_keyword:
    st.info(f"선택된 키워드: {selected_keyword}")

    keyword_articles = filter_by_keyword(df, selected_keyword)

    st.write(f"'{selected_keyword}' 관련 기사 수: {len(keyword_articles)}건")

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
        if col and col in keyword_articles.columns
    ]

    if date_col:
        keyword_articles = keyword_articles.sort_values(
            by=date_col,
            ascending=False
        )

    st.dataframe(
        keyword_articles[show_cols],
        use_container_width=True
    )

# ---------------------------
# 최신일 자동 추출 키워드
# ---------------------------

st.subheader(f"{latest_date} 자동 추출 주요 단어")

auto_keyword_df = extract_simple_words(latest_df, top_n=20)

st.dataframe(auto_keyword_df, use_container_width=True)
st.bar_chart(auto_keyword_df.set_index("keyword"))

# ---------------------------
# 주제별 기사 분석
# ---------------------------

st.subheader("주제별 기사 수")

topic_map = {
    "AI/인공지능": ["AI", "인공지능", "생성형AI", "챗GPT"],
    "반도체": ["반도체", "삼성전자", "SK하이닉스", "칩"],
    "클라우드": ["클라우드", "AWS", "Azure", "구글 클라우드"],
    "보안": ["보안", "해킹", "개인정보", "랜섬웨어"],
    "모빌리티": ["전기차", "자율주행", "배터리", "로봇"],
    "플랫폼": ["네이버", "카카오", "구글", "애플", "메타"]
}

topic_result = []

for topic, keywords in topic_map.items():
    topic_df = filter_by_topic(df, keywords)

    topic_result.append({
        "topic": topic,
        "count": len(topic_df)
    })

topic_df = pd.DataFrame(topic_result).sort_values(
    "count",
    ascending=False
)

st.dataframe(topic_df, use_container_width=True)
st.bar_chart(topic_df.set_index("topic"))

selected_topic = st.selectbox(
    "주제별 기사 상세보기",
    topic_df["topic"].tolist()
)

selected_topic_keywords = topic_map[selected_topic]
selected_topic_df = filter_by_topic(df, selected_topic_keywords)

st.write(f"{selected_topic} 관련 기사 수: {len(selected_topic_df)}건")

topic_show_cols = [
    col for col in [
        date_col,
        "media_domain",
        "source",
        "title",
        "description",
        "originallink",
        "link"
    ]
    if col and col in selected_topic_df.columns
]

if date_col:
    selected_topic_df = selected_topic_df.sort_values(
        by=date_col,
        ascending=False
    )

st.dataframe(
    selected_topic_df[topic_show_cols],
    use_container_width=True
)

# ---------------------------
# 키워드 동시 등장 상관분석
# ---------------------------

st.subheader("키워드 동시 등장 상관분석")

corr_df = pd.DataFrame(index=df.index)
text_series = make_text_series(df)

for keyword in main_keywords:
    corr_df[keyword] = text_series.str.contains(
        keyword,
        case=False,
        regex=False
    ).astype(int)

keyword_corr = corr_df.corr()

st.write("값이 1에 가까울수록 두 키워드가 같은 기사에 함께 등장하는 경향이 강함.")

st.dataframe(
    keyword_corr.style.background_gradient(axis=None),
    use_container_width=True
)

# ---------------------------
# 언론사별 기사 수
# ---------------------------

if "media_domain" in df.columns:
    st.subheader("언론사별 기사 수")

    media_count = (
        df["media_domain"]
        .fillna("unknown")
        .value_counts()
        .reset_index()
    )

    media_count.columns = ["media_domain", "count"]

    st.dataframe(media_count, use_container_width=True)
    st.bar_chart(media_count.set_index("media_domain"))

# ---------------------------
# 날짜별 기사 수
# ---------------------------

if date_col:
    st.subheader("날짜별 기사 수")

    date_count = (
        df[date_col]
        .fillna("unknown")
        .value_counts()
        .sort_index()
        .reset_index()
    )

    date_count.columns = ["date", "count"]

    st.dataframe(date_count, use_container_width=True)
    st.line_chart(date_count.set_index("date"))

# ---------------------------
# 전체 기사 검색
# ---------------------------

st.subheader("전체 기사 검색")

search_text = st.text_input("검색어를 입력하세요")

if search_text:
    result_df = filter_by_keyword(df, search_text)

    st.write(f"검색 결과: {len(result_df)}건")

    search_show_cols = [
        col for col in [
            date_col,
            "media_domain",
            "source",
            "title",
            "description",
            "originallink",
            "link"
        ]
        if col and col in result_df.columns
    ]

    if date_col:
        result_df = result_df.sort_values(
            by=date_col,
            ascending=False
        )

    st.dataframe(
        result_df[search_show_cols],
        use_container_width=True
    )

# ---------------------------
# 원본 데이터 / 파일 목록
# ---------------------------

with st.expander("원본 데이터 보기"):
    st.dataframe(df, use_container_width=True)

with st.expander("불러온 파일 목록"):
    for file in filtered_files:
        st.write(file["Key"])
