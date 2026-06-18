# IT News Analysis Dashboard

## Project Overview

본 프로젝트는 국내 주요 IT 뉴스 데이터를 자동 수집하고 데이터 마이닝 기법을 활용하여 IT 기술 트렌드를 분석하는 대시보드를 구축하는 것을 목표로 한다.

뉴스 데이터를 수집 및 전처리한 후 Streamlit 기반 대시보드를 통해 핵심 기술 키워드, 기술 간 연관성, 급부상 이슈 등을 시각화하였다.

---

## Project Objectives

본 프로젝트는 다음과 같은 질문에 답하기 위해 수행되었다.

* 현재 IT 뉴스에서 주요 기술 이슈와 핵심 키워드는 무엇인가?
* IT 기술들은 어떤 관계를 가지고 함께 언급되는가?
* 최근 IT 뉴스에서 새롭게 주목받는 기술은 무엇인가?

---

## System Architecture

총 16개의 IT 뉴스 채널을 대상으로 데이터를 수집하였다.

수집은 Amazon EventBridge를 통해 1시간마다 자동 실행되며, AWS Lambda를 이용하여 뉴스 데이터를 수집 및 전처리하였다.

전처리된 데이터는 Amazon S3에 저장되며 Streamlit Dashboard를 통해 분석 결과를 시각화하였다.

```text
News Sources
      ↓
Amazon EventBridge
      ↓
News Collector Lambda
      ↓
Amazon S3 (Raw Data)
      ↓
News Preprocessor Lambda
      ↓
Amazon S3 (Processed Data)
      ↓
Streamlit Dashboard
```

---

## Data Collection

### Collection Sources

총 16개의 뉴스 수집 채널을 활용하였다.

#### Naver API

* Naver News API

#### RSS

* DataNet
* ETNews Today
* ETNews SW
* ETNews AI

#### Google News RSS

* ZDNet Korea
* Digital Daily
* ITWorld Korea
* IT News
* Daum IT
* Digital Today
* AI Times
* BoanNews
* CIO Korea
* TechWorld
* 기타 IT 전문 매체

---

### Collection Period

```text
2026-05-14 ~ Present
```

---

### Collection Cycle

```text
Every 1 Hour
```

---

### Storage Location

#### Raw Data

```text
S3
└── it_news/IT/raw/YYYY/MM/DD/HH
```

#### Processed Data

```text
S3
└── it_news/IT/processed
```

---

## Data Preprocessing

수집된 뉴스 데이터에 대해 다음과 같은 전처리를 수행하였다.

* HTML Tag 제거
* 특수문자 제거
* 중복 기사 제거
* 날짜 형식 정규화
* 결측치 처리
* 기초 불용어 제거
* 기사 메타데이터 정리

전처리 결과는 CSV 형태로 저장하여 분석에 활용하였다.

---

## Dashboard Features

### 1. IT News Briefing

선택한 기간(일/주/월/년)의 주요 IT 뉴스 이슈를 요약하여 제공한다.

분석 내용

* 기사 수
* 핵심 키워드
* 리스크 기사 비율
* 주요 IT 이슈 요약

---

### 2. TF-IDF Based Keyword Analysis

TF-IDF(Term Frequency - Inverse Document Frequency)를 활용하여 특정 기간을 대표하는 핵심 기술 키워드를 분석한다.

분석 목적

* 단순 빈도 이상의 핵심 키워드 탐색
* 특정 시기를 특징짓는 기술 키워드 식별

예시

* AI
* 반도체
* 클라우드
* 보안
* GPU
* HBM

---

### 3. Co-occurrence Analysis

뉴스 기사 내에서 함께 등장하는 기술 키워드를 분석하여 기술 간 연관성을 파악한다.

분석 예시

* AI ↔ 디지털
* AI ↔ 반도체
* AI ↔ 보안
* 클라우드 ↔ 데이터센터

분석 목적

* 기술 간 관계 분석
* 기술 융합 트렌드 파악

---

### 4. Time Series Analysis

직전 기간 대비 증가 또는 감소한 기술 키워드를 분석한다.

분석 내용

* 증가 키워드
* 감소 키워드
* 신규 등장 키워드
* 변화율 분석

분석 목적

* 최근 급부상 이슈 탐지
* 기술 관심도 변화 파악

---

### 5. Similar Article Analysis

TF-IDF 기반 벡터화를 수행한 후 Cosine Similarity를 활용하여 기사 간 유사도를 분석한다.

분석 목적

* 유사 기사 탐색
* 중복 기사 식별
* 관련 기사 그룹 분석

---

## Applied Data Mining Techniques

| Technique              | Description   |
| ---------------------- | ------------- |
| TF-IDF                 | 핵심 키워드 추출     |
| Co-occurrence Analysis | 기술 간 연관성 분석   |
| Time Series Analysis   | 기간별 키워드 변화 분석 |
| Cosine Similarity      | 기사 간 유사도 분석   |

---

## Project Structure

```text
.
├── streamlit_app.py
│
├── source/
│   ├── news_collector_lambda.py
│   ├── news_preprocessor_lambda.py
│   └── ...
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
│
└── README.md
```

---

## How to Run

### 1. Clone Repository

```bash
git clone <repository_url>
cd data-mining
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. AWS Resources

본 프로젝트는 AWS 환경을 사용한다.

필수 구성 요소

* Amazon EventBridge
* AWS Lambda
* Amazon S3

Lambda Source Code는 `source` 폴더에 포함되어 있다.

```text
source/
├── news_collector_lambda.py
├── news_preprocessor_lambda.py
└── ...
```

---

### 4. Execution Flow

```text
1. EventBridge 실행
2. News Collector Lambda 실행
3. Raw Data 저장 (S3)
4. News Preprocessor Lambda 실행
5. Processed Data 저장 (S3)
6. Streamlit Dashboard 실행
```

---

### 5. Run Streamlit

프로젝트 루트 경로에서 실행한다.

```bash
streamlit run streamlit_app.py
```

---

### 6. Access Dashboard

```text
http://localhost:8501
```

---

## Future Improvements

향후 다음과 같은 기능을 추가할 계획이다.

### 1. 데이터 수집 기간 확대

장기적인 기술 트렌드 분석을 위해 데이터 수집 기간을 확대할 예정이다.

### 2. 기사 유사도 분석 고도화

유사 기사 클러스터링 기능을 추가하여 관련 기사 그룹을 자동으로 분류할 계획이다.

### 3. BERT 기반 감성 분석

기사에 대한 긍정·부정 감성 분석을 적용하여 기술 이슈에 대한 시장 반응을 분석할 계획이다.

### 4. 한국어 형태소 분석 적용

KoNLPy, Mecab 등을 활용하여 명사 중심의 키워드 추출 정확도를 향상시킬 계획이다.

---

## Technology Stack

### Cloud

* AWS Lambda
* Amazon EventBridge
* Amazon S3

### Backend

* Python

### Data Processing

* Pandas
* NumPy
* Scikit-learn

### Visualization

* Streamlit

---

## Author

Data Mining Project

2026
