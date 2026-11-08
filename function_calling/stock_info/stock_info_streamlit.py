import json  # GPT가 JSON을 반환할 때, JSON을 파싱하기 위해 필요
import os

import streamlit as st
from dotenv import load_dotenv
from gpt_function import get_current_time, get_yf_stock_info, tools
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


# 1 7 0
def get_ai_response(messages, tools=None):
    response = client.chat.completions.create(
        model="gpt-6-luna",
        messages=messages,  # 대화 기록을 입력으로 전달
        tools=tools,  # 사용 가능한 도구 목록 전달
        reasoning_effort="none",  # gpt-6-luna에서 Chat Completions로 함수 도구를 쓰려면 reasoning_effort="none"을 명시해야 해.
    )
    return response


st.title("AI 상담사와 대화하기")

# 세션에 대화 기록이 없을 때만 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "너는 사용자를 도와주는 상담사야.",
        }
    ]

# 저장된 대화 중 사용자 질문과 AI 답변만 화면에 표시
# system 지시문과 tool 실행 결과는 화면에 표시하지 않음
for msg in st.session_state.messages:
    if msg["role"] in ("user", "assistant") and msg.get("content"):
        st.chat_message(msg["role"]).write(msg["content"])

# 사용자가 입력을 제출한 경우에만 아래 코드 실행
# 입력하지 않았으면 None이므로 이 블록을 건너뜀
if user_input := st.chat_input("메시지를 입력해 주세요"):
    # 사용자 메시지는 한 번만 저장하고 화면에 표시
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )
    st.chat_message("user").write(user_input)

    with st.spinner("답변을 준비하고 있어요..."):
        # 함수 호출과 결과 전달을 반복하는 안쪽 반복문은 유지
        while True:
            ai_response = get_ai_response(
                st.session_state.messages,
                tools=tools,
            )
            ai_message = ai_response.choices[0].message

            # AI의 함수 호출 요청을 함수 실행 결과보다 먼저 기록
            st.session_state.messages.append(ai_message.model_dump(exclude_none=True))

            # 함수 호출이 없으면 최종 답변을 웹 화면에 표시
            if not ai_message.tool_calls:
                st.chat_message("assistant").write(ai_message.content or "")
                break

            # AI가 요청한 모든 함수 호출을 처리
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call.function.name

                try:
                    # JSON 문자열을 파이썬 딕셔너리로 변환
                    arguments = json.loads(tool_call.function.arguments)

                    if tool_name == "get_current_time":
                        result = get_current_time(timezone=arguments.get("timezone"))
                    elif tool_name == "get_yf_stock_info":
                        result = get_yf_stock_info(ticker=arguments.get("ticker"))
                    else:
                        result = f"지원하지 않는 함수: {tool_name}"

                except Exception as error:
                    # 함수 실행이 실패해도 해당 호출의 결과를 기록
                    result = f"함수 실행 실패: {error}"

                # 호출 ID로 AI의 요청과 실행 결과를 연결
                st.session_state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

            # 다음 반복에서 함수 결과를 AI에 전달해 답변 생성
