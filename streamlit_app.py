import streamlit as st
import pandas as pd
import boto3
import re
from collections import Counter

st.set_page_config(
    page_title="IT 뉴스 분석 대시보드",
    layout="wide"
)

st.title("IT 뉴스 분석 대시보드")

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


def make_text_column(df):
    text = ""

    for col in ["title", "description"]:
        if col in df.columns:
            text += " " + df[col].fillna("").astype(str)

    return text


def count_keywords(df, keywords):
    text_series = make_text_column(df)

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
    text = " ".join(make_text_column(df).tolist())

    words = re.findall(r"[가-힣A-Za-z0-9]{2,}", text)

    stopwords = {
        "기자", "뉴스", "오늘", "이번", "대한", "관련", "통해", "위해",
        "있는", "했다", "한다", "지난", "오는", "올해", "기업", "산업",
        "서비스", "시장", "기술", "제공", "발표", "추진", "구축", "사용",
        "naver", "google"
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

# 중복 제거
dedup_cols = []

for col in ["originallink", "link", "title"]:
    if col in df.columns:
        dedup_cols.append(col)

if dedup_cols:
    df = df.drop_duplicates(subset=dedup_cols, keep="last")

if "pubDate_ymd" in df.columns:
    df["pubDate_ymd"] = df["pubDate_ymd"].astype(str)
    latest_date = df["pubDate_ymd"].max()
else:
    latest_date = "unknown"

latest_df = df[df["pubDate_ymd"] == latest_date].copy()


# ---------------------------
# 분석 키워드
# ---------------------------

main_keywords = [
    "AI",
    "인공지능",
    "생성형AI",
    "반도체",
    "클라우드",
    "보안",
    "데이터",
    "로봇",
    "배터리",
    "전기차",
    "챗GPT",
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
    else:
        st.metric("수집 소스 수", df["source"].nunique())

with col4:
    st.metric("수집 파일 수", len(filtered_files))

st.write("최신일 주요 키워드 TOP 10")
st.bar_chart(top_keywords.set_index("keyword"))


# ---------------------------
# 키워드 클릭형 필터
# ---------------------------

st.subheader("키워드 클릭해서 기사 보기")

if "selected_keyword" not in st.session_state:
    st.session_state["selected_keyword"] = top_keywords.iloc[0]["keyword"]

button_cols = st.columns(5)

for idx, row in top_keywords.reset_index(drop=True).iterrows():
    keyword = row["keyword"]
    count = row["count"]

    with button_cols[idx % 5]:
        if st.button(f"{keyword} ({count})"):
            st.session_state["selected_keyword"] = keyword

selected_keyword = st.session_state["selected_keyword"]

st.info(f"선택된 키워드: {selected_keyword}")

keyword_articles = filter_by_keyword(df, selected_keyword)

st.write(f"'{selected_keyword}' 관련 기사 수: {len(keyword_articles)}건")

show_cols = [
    col for col in [
        "pubDate_ymd",
        "media_domain",
        "source",
        "title",
        "description",
        "originallink",
        "link"
    ]
    if col in keyword_articles.columns
]

st.dataframe(
    keyword_articles[show_cols].sort_values(
        by="pubDate_ymd",
        ascending=False
    ),
    use_container_width=True
)


# ---------------------------
# 최신일 주요 키워드 자동 추출
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
    topic_condition = pd.Series(False, index=df.index)

    for keyword in keywords:
        topic_condition = topic_condition | filter_by_keyword(df, keyword).index.to_series().isin(df.index)

    topic_result.append({
        "topic": topic,
        "count": int(topic_condition.sum())
    })

topic_df = pd.DataFrame(topic_result).sort_values("count", ascending=False)

st.dataframe(topic_df, use_container_width=True)
st.bar_chart(topic_df.set_index("topic"))


# ---------------------------
# 키워드 상관분석
# ---------------------------

st.subheader("키워드 동시 등장 상관분석")

corr_df = pd.DataFrame(index=df.index)

text_series = make_text_column(df)

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

if "pubDate_ymd" in df.columns:
    st.subheader("날짜별 기사 수")

    date_count = (
        df["pubDate_ymd"]
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

    st.dataframe(
        result_df[show_cols].sort_values(
            by="pubDate_ymd",
            ascending=False
        ),
        use_container_width=True
    )


# ---------------------------
# 원본 데이터
# ---------------------------

with st.expander("원본 데이터 보기"):
    st.dataframe(df, use_container_width=True)

with st.expander("불러온 파일 목록"):
    for file in filtered_files:
        st.write(file["Key"])
