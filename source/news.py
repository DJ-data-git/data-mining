import os
import json
import csv
import io
import urllib.parse
import urllib.request
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

import boto3

s3 = boto3.client("s3")

# 환경변수
NAVER_CLIENT_ID = os.environ["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = os.environ["NAVER_CLIENT_SECRET"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
PREFIX = os.environ.get("PREFIX", "news")


def clean_html(text):
    """
    HTML 태그 제거
    """
    return re.sub(r"<.*?>", "", text or "").strip()


def format_pubdate_ymd(date_str):
    """
    RFC2822 형식 날짜를 YYYY-MM-DD로 변환
    예: Wed, 22 Apr 2026 11:13:08 +0900 -> 2026-04-22
    """
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return date_str


# ---------------------------
# 네이버 뉴스 API
# ---------------------------
def fetch_naver_news(query: str, display: int = 100, start: int = 1, sort: str = "date"):
    base_url = "https://openapi.naver.com/v1/search/news.json"
    params = {
        "query": query,
        "display": display,
        "start": start,
        "sort": sort
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
    req.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)

    with urllib.request.urlopen(req) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def fetch_all_naver_news(query: str, sort: str = "date", max_items: int = 1000):
    """
    네이버 뉴스 최대 1000건 수집
    """
    all_items = []
    first_meta = None
    start = 1

    while start <= 1000 and len(all_items) < max_items:
        remaining = max_items - len(all_items)
        display = min(100, remaining)

        result = fetch_naver_news(query=query, display=display, start=start, sort=sort)

        if first_meta is None:
            first_meta = {
                "lastBuildDate": result.get("lastBuildDate", ""),
                "total": result.get("total", ""),
                "start": result.get("start", ""),
                "display": result.get("display", "")
            }

        items = result.get("items", [])
        if not items:
            break

        normalized_items = []
        for item in items:
            normalized_items.append({
                "source": "naver",
                "title": clean_html(item.get("title", "")),
                "originallink": item.get("originallink", ""),
                "link": item.get("link", ""),
                "description": clean_html(item.get("description", "")),
                "pubDate": item.get("pubDate", ""),
                "pubDate_ymd": format_pubdate_ymd(item.get("pubDate", ""))
            })

        all_items.extend(normalized_items)

        if len(items) < display:
            break

        start += display
        time.sleep(0.1)

    return all_items[:max_items], first_meta or {}


# ---------------------------
# 구글 뉴스 RSS
# ---------------------------
def fetch_google_news_rss(query: str, hl: str = "ko", gl: str = "KR", ceid: str = "KR:ko"):
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl={hl}&gl={gl}&ceid={ceid}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(req) as response:
        return response.read()


def fetch_all_google_news(query: str, max_items: int = 100):
    """
    구글 뉴스 RSS 수집
    RSS 특성상 실제 건수는 max_items보다 적을 수 있음
    """
    xml_data = fetch_google_news_rss(query)
    root = ET.fromstring(xml_data)

    channel = root.find("channel")
    if channel is None:
        return [], {}

    rss_items = channel.findall("item")
    all_items = []

    for item in rss_items[:max_items]:
        title = item.findtext("title", default="")
        link = item.findtext("link", default="")
        description = item.findtext("description", default="")
        pub_date = item.findtext("pubDate", default="")

        all_items.append({
            "source": "google",
            "title": clean_html(title),
            "originallink": link,
            "link": link,
            "description": clean_html(description),
            "pubDate": pub_date,
            "pubDate_ymd": format_pubdate_ymd(pub_date)
        })

    meta = {
        "lastBuildDate": channel.findtext("lastBuildDate", default=""),
        "total": len(all_items),
        "start": 1,
        "display": len(all_items)
    }

    return all_items, meta


# ---------------------------
# S3 저장
# ---------------------------
def save_json_to_s3(data: dict, bucket: str, key: str):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json; charset=utf-8"
    )


def save_csv_to_s3(items: list, bucket: str, key: str, meta: dict):
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "source",
        "lastBuildDate",
        "total",
        "start",
        "display",
        "title",
        "originallink",
        "link",
        "description",
        "pubDate",
        "pubDate_ymd"
    ])

    for item in items:
        writer.writerow([
            item.get("source", ""),
            meta.get("lastBuildDate", ""),
            meta.get("total", ""),
            meta.get("start", ""),
            meta.get("display", ""),
            item.get("title", ""),
            item.get("originallink", ""),
            item.get("link", ""),
            item.get("description", ""),
            item.get("pubDate", ""),
            item.get("pubDate_ymd", "")
        ])

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=output.getvalue().encode("utf-8-sig"),
        ContentType="text/csv; charset=utf-8"
    )


# ---------------------------
# Lambda Handler
# ---------------------------
def lambda_handler(event, context):
    """
    event 예시:
    {
      "query": "IT",
      "source": "all",
      "sort": "date",
      "naver_max_items": 1000,
      "google_max_items": 100
    }

    source:
      - naver
      - google
      - all
    """

    query = event.get("query", "IT")
    source = event.get("source", "all").lower()
    sort = event.get("sort", "date")

    # 네이버/구글 분리
    naver_max_items = int(event.get("naver_max_items", 1000))
    google_max_items = int(event.get("google_max_items", 100))

    # 네이버는 API 한도상 최대 1000
    naver_max_items = min(naver_max_items, 1000)

    # 한국시간 기준
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    date_path = now.strftime("%Y/%m/%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    all_items = []
    meta_by_source = {}

    # 네이버 수집
    if source in ["naver", "all"]:
        naver_items, naver_meta = fetch_all_naver_news(
            query=query,
            sort=sort,
            max_items=naver_max_items
        )
        all_items.extend(naver_items)
        meta_by_source["naver"] = naver_meta

    # 구글 수집
    if source in ["google", "all"]:
        google_items, google_meta = fetch_all_google_news(
            query=query,
            max_items=google_max_items
        )
        all_items.extend(google_items)
        meta_by_source["google"] = google_meta

    # 전체 메타
    combined_meta = {
        "query": query,
        "source": source,
        "collected_at": now.isoformat(),
        "item_count": len(all_items),
        "meta_by_source": meta_by_source
    }

    # 전체 CSV용 메타
    csv_meta = {
        "lastBuildDate": now.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(all_items),
        "start": 1,
        "display": len(all_items)
    }

    # S3 key
    safe_query = query.replace(" ", "_")
    json_key = f"{PREFIX}/{safe_query}/{date_path}/news_{source}_{timestamp}.json"
    csv_key = f"{PREFIX}/{safe_query}/{date_path}/news_{source}_{timestamp}.csv"

    # 저장
    save_json_to_s3(
        {
            "meta": combined_meta,
            "items": all_items
        },
        BUCKET_NAME,
        json_key
    )

    save_csv_to_s3(
        all_items,
        BUCKET_NAME,
        csv_key,
        csv_meta
    )

    return {
        "statusCode": 200,
        "message": "Saved news data to S3 successfully",
        "query": query,
        "source": source,
        "naver_max_items": naver_max_items,
        "google_max_items": google_max_items,
        "item_count": len(all_items),
        "meta_by_source": meta_by_source,
        "json_key": json_key,
        "csv_key": csv_key
    }
