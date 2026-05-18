import streamlit as st
import pandas as pd
import boto3
import re

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

    csv_files = [
        file for file in all_files
        if file["Key"].endswith(".csv")
    ]

    return csv_files


csv_files = list_all_csv_files(BUCKET_NAME, PREFIX)

filtered_files = []

for file in csv_files:
    key = file["Key"]
    file_date = extract_yyyymmdd_from_key(key)

    if file_date and file_date >= START_DATE:
        filtered_files.append(file)

if not filtered_files:
    st.error("2026년 5월 14일 이후 CSV 파일을 찾지 못했습니다.")
    st.stop()

filtered_files = sorted(
    filtered_files,
    key=lambda x: x["Key"]
)

st.info(f"불러온 CSV 파일 수: {len(filtered_files)}개")

with st.expander("불러온 파일 목록"):
    for file in filtered_files:
        st.write(file["Key"])

df_list = []

for file in filtered_files:
    key = file["Key"]

    obj = s3.get_object(
        Bucket=BUCKET_NAME,
        Key=key
    )

    temp_df = pd.read_csv(obj["Body"])
    temp_df["loaded_file"] = key

    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# 중복 제거
dedup_key_cols = []

for col in ["originallink", "link", "title"]:
    if col in df.columns:
        dedup_key_cols.append(col)

if dedup_key_cols:
    df = df.drop_duplicates(subset=dedup_key_cols, keep="last")

st.subheader("기본 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("전체 기사 수", len(df))

with col2:
    st.metric("불러온 파일 수", len(filtered_files))

with col3:
    if "media_domain" in df.columns:
        st.metric("언론사 수", df["media_domain"].nunique())
    elif "source" in df.columns:
        st.metric("수집 소스 수", df["source"].nunique())
    else:
        st.metric("언론사 수", "확인 불가")

with col4:
    if "pubDate_ymd" in df.columns:
        st.metric("날짜 수", df["pubDate_ymd"].nunique())
    else:
        st.metric("날짜 수", "확인 불가")

st.write("컬럼 목록:", list(df.columns))

st.subheader("원본 데이터")
st.dataframe(df, use_container_width=True)

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
    st.bar_chart(source_count.set_index("source"))

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
st.bar_chart(keyword_df.set_index("keyword"))

st.subheader("기사 검색")

search_text = st.text_input("검색어를 입력하세요")

if search_text:
    search_df = df.copy()

    condition = pd.Series(False, index=search_df.index)

    if "title" in search_df.columns:
        condition = condition | search_df["title"].fillna("").str.contains(
            search_text,
            case=False,
            regex=False
        )

    if "description" in search_df.columns:
        condition = condition | search_df["description"].fillna("").str.contains(
            search_text,
            case=False,
            regex=False
        )

    result_df = search_df[condition]

    st.write(f"검색 결과: {len(result_df)}건")
    st.dataframe(result_df, use_container_width=True)
