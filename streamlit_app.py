import streamlit as st
import pandas as pd
import boto3

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
# S3 CSV 경로
# ---------------------------

PREFIX = "it_news/IT/processed/"

# ---------------------------
# S3에서 최신 CSV 파일 찾기
# ---------------------------

response = s3.list_objects_v2(
    Bucket=BUCKET_NAME,
    Prefix=PREFIX
)

files = response.get("Contents", [])

csv_files = [
    file for file in files
    if file["Key"].endswith(".csv")
]

if not csv_files:
    st.error("S3에서 CSV 파일을 찾지 못했습니다.")
    st.stop()

latest_file = sorted(
    csv_files,
    key=lambda x: x["LastModified"],
    reverse=True
)[0]

latest_key = latest_file["Key"]

st.info(f"불러온 파일: {latest_key}")

# ---------------------------
# 최신 CSV 읽기
# ---------------------------

obj = s3.get_object(
    Bucket=BUCKET_NAME,
    Key=latest_key
)

df = pd.read_csv(obj["Body"])

# ---------------------------
# 기본 데이터 확인
# ---------------------------

st.subheader("기본 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 기사 수", len(df))

with col2:
    st.metric("컬럼 수", len(df.columns))

with col3:
    if "media_domain" in df.columns:
        st.metric("언론사 수", df["media_domain"].nunique())
    elif "source" in df.columns:
        st.metric("수집 소스 수", df["source"].nunique())
    else:
        st.metric("언론사 수", "확인 불가")

st.write("컬럼 목록:", list(df.columns))

# ---------------------------
# 원본 데이터
# ---------------------------

st.subheader("원본 데이터")
st.dataframe(df, use_container_width=True)

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

    st.bar_chart(
        media_count.set_index("media_domain")
    )

elif "source" in df.columns:
    st.subheader("수집 소스별 기사 수")

    source_count = (
        df["source"]
        .fillna("unknown")
        .value_counts()
        .reset_index()
    )

    source_count.columns = ["source", "count"]

    st.dataframe(source_count, use_container_width=True)

    st.bar_chart(
        source_count.set_index("source")
    )

else:
    st.warning("media_domain 또는 source 컬럼이 없습니다.")

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

    st.line_chart(
        date_count.set_index("date")
    )
else:
    st.warning("pubDate_ymd 컬럼이 없습니다.")

# ---------------------------
# 키워드 분석
# ---------------------------

st.subheader("키워드별 기사 수")

keywords = [
    "AI",
    "인공지능",
    "생성형AI",
    "반도체",
    "클라우드",
    "보안",
    "데이터",
    "로봇",
    "배터리",
    "전기차"
]

keyword_result = []

for keyword in keywords:
    count = 0

    if "title" in df.columns:
        count += df["title"].fillna("").str.contains(
            keyword,
            case=False,
            regex=False
        ).sum()

    if "description" in df.columns:
        count += df["description"].fillna("").str.contains(
            keyword,
            case=False,
            regex=False
        ).sum()

    keyword_result.append({
        "keyword": keyword,
        "count": int(count)
    })

keyword_df = pd.DataFrame(keyword_result)

st.dataframe(keyword_df, use_container_width=True)

st.bar_chart(
    keyword_df.set_index("keyword")
)

# ---------------------------
# 기사 검색
# ---------------------------

st.subheader("기사 검색")

search_text = st.text_input("검색어를 입력하세요")

if search_text:
    search_df = df.copy()

    condition = False

    if "title" in search_df.columns:
        condition = search_df["title"].fillna("").str.contains(
            search_text,
            case=False,
            regex=False
        )

    if "description" in search_df.columns:
        desc_condition = search_df["description"].fillna("").str.contains(
            search_text,
            case=False,
            regex=False
        )

        if isinstance(condition, bool):
            condition = desc_condition
        else:
            condition = condition | desc_condition

    result_df = search_df[condition]

    st.write(f"검색 결과: {len(result_df)}건")
    st.dataframe(result_df, use_container_width=True)
