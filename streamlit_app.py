import streamlit as st
import pandas as pd
import boto3

# ---------------------------
# 기본 설정
# ---------------------------

st.set_page_config(
    page_title="IT 뉴스 분석 대시보드",
    layout="wide"
)

st.title("IT 뉴스 분석 대시보드")

# ---------------------------
# Streamlit Secrets에서 AWS 정보 가져오기
# ---------------------------

AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = st.secrets["AWS_REGION"]
BUCKET_NAME = st.secrets["BUCKET_NAME"]

# ---------------------------
# S3 클라이언트 생성
# ---------------------------

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# ---------------------------
# S3에서 최신 CSV 파일 찾기
# ---------------------------

PREFIX = "news/IT/"

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

st.write("불러온 파일:", latest_key)

# ---------------------------
# 최신 CSV 읽기
# ---------------------------

obj = s3.get_object(
    Bucket=BUCKET_NAME,
    Key=latest_key
)

df = pd.read_csv(obj["Body"])

# ---------------------------
# 데이터 출력
# ---------------------------

st.subheader("원본 데이터")
st.dataframe(df)

st.subheader("기본 정보")
st.write("전체 기사 수:", len(df))
st.write("컬럼 목록:", list(df.columns))

# ---------------------------
# 언론사별 기사 수
# ---------------------------

if "media_domain" in df.columns:
    st.subheader("언론사별 기사 수")

    media_count = df["media_domain"].value_counts().reset_index()
    media_count.columns = ["media_domain", "count"]

    st.dataframe(media_count)
    st.bar_chart(
        media_count.set_index("media_domain")
    )
else:
    st.warning("media_domain 컬럼이 없습니다.")
