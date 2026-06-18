# IT News Analysis Dashboard

## Project Overview

본 프로젝트는 국내 주요 IT 뉴스 데이터를 수집하여 데이터 마이닝 기법을 활용한 분석 대시보드를 구축하는 것을 목표로 한다.

뉴스 데이터를 자동 수집 및 전처리한 후, Streamlit 기반 대시보드를 통해 주요 IT 기술 이슈, 기술 간 연관성, 급부상 이슈 등을 시각화하였다.

---

## Data Collection

### Collection Sources

총 16개의 IT 뉴스 채널을 대상으로 데이터를 수집하였다.

* Naver News API
* Google News RSS
* DataNet
* ETNews
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

### Collection Period

* 2026-05-14 ~ Present

### Collection Cycle

* Amazon EventBridge
* Every 1 Hour

---

## System Architecture

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

## Data Preprocessing

수집된 뉴스 데이터에 대해 다음 전처리를 수행하였다.

* HTML Tag 제거
* 특수문자 제거
* 중복 기사 제거
* 날짜 형식 정규화
* 불용어 제거
* 기사 메타데이터 정리

전처리 결과는 CSV 형식으로 저장하여 분석에 활용하였다.

---

## Dashboard Features

### 1. IT News Briefing

선택한 기간(일/주/월/년)의 주요 IT 뉴스 이슈를 요약하여 제공한다.

분석 내용

* 기사 수
* 핵심 키워드
* 리스크 기사 비율
* 주요 기술 이슈

---

### 2. TF-IDF Based Keyword Analysis

TF-IDF(Term Frequency - Inverse Document Frequency)를 활용하여 특정 기간을 대표하는 핵심 기술 키워드를 분석한다.

분석 결과 예시

* AI
* 반도체
* 클라우드
* 보안
* 데이터

---

### 3. Co-occurrence Analysis

뉴스 기사 내에서 함께 등장하는 기술 키워드를 분석하여 기술 간 연관성을 파악한다.

예시

* AI ↔ 디지털
* AI ↔ 반도체
* 클라우드 ↔ 보안

---

### 4. Time Series Analysis

직전 기간 대비 증가하거나 감소한 기술 키워드를 분석한다.

분석 내용

* 증가 키워드
* 감소 키워드
* 신규 등장 키워드
* 변화율

이를 통해 최근 급부상한 IT 이슈를 탐지할 수 있다.

---

### 5. Similar Article Analysis

TF-IDF 기반 기사 벡터화를 수행하고 유사 기사 분석을 통해 관련 기사 그룹을 탐색한다.

---

## Applied Data Mining Techniques

본 프로젝트에서 활용한 주요 데이터 마이닝 기법은 다음과 같다.

| Technique              | Description   |
| ---------------------- | ------------- |
| TF-IDF                 | 핵심 키워드 추출     |
| Co-occurrence Analysis | 기술 간 연관성 분석   |
| Time Series Analysis   | 기간별 키워드 변화 분석 |
| Cosine Similarity      | 기사 간 유사도 분석   |

---

## Future Improvements

향후 다음과 같은 기능을 추가할 계획이다.

* 데이터 수집 기간 확대
* 기사 유사도 분석 고도화
* BERT 기반 감성 분석 적용
* 한국어 형태소 분석 적용
* 기술 트렌드 예측 기능 추가

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
* Scikit-Learn

### Visualization

* Streamlit

---

## Author

Data Mining Project

2026
