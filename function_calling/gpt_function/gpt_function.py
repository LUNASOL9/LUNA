from datetime import datetime

import pytz  # python -m pip install pytz


def get_current_time(timezone: str = "Asia/Seoul"):
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    now_timezone = f"({now} {timezone})"
    print(now_timezone)

    return now_timezone


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
]


if __name__ == "__main__":
    get_current_time("America/New_York")
