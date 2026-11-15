from datetime import datetime

import pytz  # python -m pip install pytz
import yfinance as yf  # python -m pip install yfinance


def get_current_time(timezone: str = "Asia/Seoul"):
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    now_timezone = f"({now} {timezone})"
    print(now_timezone)

    return now_timezone


# 티커를 기반으로 Yahoo Finance에서 주식 정보를 가져오는 함수
def get_yf_stock_info(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    print(info)

    return str(info)


# 티커와 기간을 기반으로 Yahoo Finance에서 주식의 과거 데이터를 가져오는 함수
def get_yf_stock_history(ticker: str, period: str = str):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    # pip install tabulate
    history_md = history.to_markdown()  # Convert DataFrame to Markdown format
    print(history)

    return history_md


# 티커를 기반으로 Yahoo Finance에서 주식 추천 정보를 가져오는 함수
def get_yf_stock_recommendations(ticker: str):
    stock = yf.Ticker(ticker)
    recommendations = stock.recommendations
    recommendations_md = (
        recommendations.to_markdown()
    )  # Convert DataFrame to Markdown format
    print(recommendations)

    return recommendations_md


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "현재 타임존의 날짜와 시간을 반환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "타임존 이름 (예: 'Asia/Seoul', 'America/New_York')",
                    },
                },
                "required": [],  # timezone은 선택 사항으로, 기본값은 'Asia/Seoul'입니다.
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_yf_stock_info",
            "description": "주식 티커를 기반으로 Yahoo Finance에서 주식 정보를 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "주식 티커 (예: 'AAPL', 'GOOGL')",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_yf_stock_history",
            "description": "주식 티커와 기간을 기반으로 Yahoo Finance에서 주식의 과거 데이터를 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "주식 티커 (예: 'AAPL', 'GOOGL')",
                    },
                    "period": {
                        "type": "string",
                        "description": "기간 (예: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')",
                    },
                },
                "required": ["ticker", "period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_yf_stock_recommendations",
            "description": "주식 티커를 기반으로 Yahoo Finance에서 주식 추천 정보를 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "주식 티커 (예: 'AAPL', 'GOOGL')",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]


# if __name__ == "__main__":
#     get_current_time("America/New_York")
#     info = get_yf_stock_info("AAPL")
#     history = get_yf_stock_history("AAPL", "1mo")
#     recommendations = get_yf_stock_recommendations("AAPL")
