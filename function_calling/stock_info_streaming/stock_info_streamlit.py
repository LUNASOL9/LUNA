import json  # JSON 문자열 ↔ 파이썬 객체 변환
import os  # 환경 변수에서 API 키 읽기

import streamlit as st
from dotenv import load_dotenv
from gpt_function import (
    get_current_time,
    get_yf_stock_history,
    get_yf_stock_info,
    get_yf_stock_recommendations,
    tools,  # AI에게 전달할 함수 이름·설명·매개변수 목록
)
from openai import OpenAI

# .env 파일의 설정을 환경 변수로 불러옴
load_dotenv()

# API 요청을 보낼 클라이언트 생성
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_ai_response(messages, tools=None, stream=True):
    """대화 기록과 도구 목록을 전달하고 AI 응답을 받는 함수."""

    # stream=True이면 완성된 답변 대신
    # 응답 조각(chunk)을 순서대로 읽을 수 있는 스트림을 반환
    #
    # 여기서는 yield를 쓰지 않고 SDK의 반환값을 그대로 전달하므로
    # stream=False일 때는 완성된 응답 객체가 반환됨
    return client.chat.completions.create(
        model="gpt-6-luna",
        messages=messages,
        tools=tools,
        reasoning_effort="none",
        stream=stream,
    )


# AI가 반환한 함수 이름을 실제 파이썬 함수와 연결하는 딕셔너리
#
# 오른쪽에 ()가 없는 이유:
# 지금 실행하는 게 아니라, 나중에 실행할 함수 자체를 저장하기 때문
FUNCTIONS = {
    "get_current_time": get_current_time,
    "get_yf_stock_info": get_yf_stock_info,
    "get_yf_stock_history": get_yf_stock_history,
    "get_yf_stock_recommendations": get_yf_stock_recommendations,
}


def execute_tool(tool_call):
    """완성된 함수 호출 정보를 받아 함수를 실행하고 결과를 반환."""

    try:
        # 중첩 딕셔너리에서 실행할 함수 이름을 꺼냄
        # 예: "get_yf_stock_info"
        name = tool_call["function"]["name"]

        # AI가 보내는 arguments는 JSON 문자열
        # 예: '{"ticker": "AAPL"}'
        #
        # json.loads()로 파이썬 딕셔너리로 변환
        # 결과: {"ticker": "AAPL"}
        arguments = json.loads(tool_call["function"]["arguments"])

        # **arguments로 전달하려면 딕셔너리여야 함
        # JSON 배열이나 숫자 등이 들어온 경우 실행하지 않음
        if not isinstance(arguments, dict):
            raise ValueError("함수 인자는 JSON 객체여야 합니다.")

        # 프로그램에서 등록한 함수인지 확인
        if name not in FUNCTIONS:
            raise ValueError(f"지원하지 않는 함수: {name}")

        # FUNCTIONS[name]으로 실제 함수를 찾음
        #
        # **arguments는 딕셔너리를 키워드 인자로 펼치는 문법
        # 예:
        # arguments = {"ticker": "AAPL", "period": "1mo"}
        #
        # 함수(**arguments)는 아래와 같은 의미
        # 함수(ticker="AAPL", period="1mo")
        #
        # 인자가 생략됐다면 해당 파이썬 함수의 기본값을 사용
        result = FUNCTIONS[name](**arguments)

        # 이미 문자열인 결과는 그대로 반환
        if isinstance(result, str):
            return result

        # 딕셔너리·리스트 등의 결과는 JSON 문자열로 변환
        # ensure_ascii=False: 한글을 그대로 표시
        # default=str: 기본 JSON 변환이 안 되는 객체는 문자열로 변환
        return json.dumps(
            result,
            ensure_ascii=False,
            default=str,
        )

    except Exception as error:
        # 인자 해석이나 함수 실행이 실패하면 오류 내용을 반환
        # 호출한 쪽은 이것도 tool 메시지로 AI에게 전달
        return json.dumps(
            {"error": str(error)},
            ensure_ascii=False,
        )


def read_stream(messages, placeholder):
    """답변을 실시간으로 표시하고 함수 호출 조각을 모으는 함수."""

    # 일반 답변 텍스트를 누적할 문자열
    content = ""

    # 거절 안내 텍스트가 들어올 경우 누적할 문자열
    refusal = ""

    # 함수 호출 조각을 호출 번호(index)별로 모을 딕셔너리
    # 여러 함수 호출이 들어와도 각각 따로 저장
    calls_by_index = {}

    # 이번 응답 생성이 끝난 이유를 저장
    finish_reason = None

    # API에 스트리밍 요청
    response = get_ai_response(
        messages,
        tools=tools,
        stream=True,
    )

    try:
        # 서버에서 응답 조각이 도착할 때마다 반복
        for chunk in response:
            # 일부 청크는 사용량 정보 등만 담고 choices가 비어 있을 수 있음
            # 비어 있을 때 [0]에 접근하면 오류가 나므로 건너뜀
            if not chunk.choices:
                continue

            # 첫 번째 응답 후보를 선택
            # 현재 코드는 한 개의 응답을 받는 설정
            choice = chunk.choices[0]

            # delta에는 이번 청크에서 새로 전달된 정보만 들어 있음
            # 전체 답변이 아니라 텍스트나 함수 호출 정보의 일부
            delta = choice.delta

            # 생성 중에는 대개 None이고, 마지막에 종료 사유가 들어옴
            # stop: 일반 응답 생성 완료
            # tool_calls: 함수 호출 요청 생성 완료
            # length: 출력 한도에 도달
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

            # 일반 답변 텍스트가 있으면 기존 문자열 뒤에 붙임
            if delta.content:
                content += delta.content

                # 같은 placeholder를 갱신하므로 새 줄을 계속 만드는 대신
                # 한 위치에서 답변이 길어지는 것처럼 보임
                # ▌는 생성 중임을 보여주는 표시용 문자
                placeholder.markdown(content + " ▌")

            # 거절 안내가 있으면 별도로 누적하고 표시
            if delta.refusal:
                refusal += delta.refusal
                placeholder.markdown(refusal)

            # 함수 호출 조각이 있으면 하나씩 처리
            #
            # delta.tool_calls가 None이면 빈 리스트 []를 사용
            # 빈 리스트는 반복할 항목이 없으므로 그냥 건너뜀
            for part in delta.tool_calls or []:
                # 이번 응답에서 처음 보는 호출 번호라면
                # 해당 호출 정보를 모을 빈 딕셔너리 생성
                if part.index not in calls_by_index:
                    calls_by_index[part.index] = {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": "",
                        },
                    }

                # 해당 호출의 딕셔너리를 짧은 이름으로 참조
                #
                # 복사본이 아니라 같은 딕셔너리를 가리키므로
                # call을 수정하면 calls_by_index 안의 내용도 바뀜
                call = calls_by_index[part.index]

                # 호출 ID가 들어온 청크에서 ID를 저장
                # 이 ID는 나중에 함수 실행 결과와 호출 요청을 연결할 때 사용
                if part.id:
                    call["id"] = part.id

                # 이번 조각에 함수 정보가 있을 때만 내부 값에 접근
                if part.function:
                    # 함수 이름 조각이 있으면 이어 붙임
                    if part.function.name:
                        call["function"]["name"] += part.function.name

                    # 인자 문자열은 여러 청크로 나뉘어 올 수 있음
                    #
                    # 예:
                    # '{"ticker":'
                    # ' "AAPL"'
                    # '}'
                    #
                    # 아직 json.loads() 하지 않고 문자열로 계속 이어 붙임
                    if part.function.arguments:
                        call["function"]["arguments"] += part.function.arguments

    finally:
        # 정상 완료되거나 읽는 중 오류가 나더라도 스트림을 닫음
        response.close()

    # 응답이 정상적인 종료 상태에 도달했는지 확인
    # 중간에 잘린 함수 호출 정보는 실행하지 않음
    if finish_reason not in ("stop", "tool_calls"):
        raise RuntimeError(f"응답이 완성되기 전에 종료됐습니다: {finish_reason}")

    # 호출 번호를 정렬한 뒤, 완성된 호출 정보만 리스트로 꺼냄
    # 예: {0: 호출A, 1: 호출B} → [호출A, 호출B]
    tool_calls = [calls_by_index[index] for index in sorted(calls_by_index)]

    # 실행에 필요한 최소한의 호출 정보가 있는지 확인
    # 인자 JSON 해석은 execute_tool()에서 수행
    for call in tool_calls:
        if not call["id"] or not call["function"]["name"]:
            raise RuntimeError("함수 호출 정보가 불완전합니다.")

    # 일반 답변이 있으면 사용하고, 없으면 거절 안내를 사용
    display_text = content or refusal

    if display_text:
        # 스트리밍 중 붙였던 ▌ 표시를 제거하고 완성된 내용 표시
        placeholder.markdown(display_text)
    else:
        # 함수 호출만 있었다면 답변 텍스트가 없을 수 있음
        placeholder.empty()

    # API에 다시 보낼 수 있는 완성된 assistant 메시지 생성
    # 함수 호출만 있을 때는 content가 None일 수 있음
    message = {
        "role": "assistant",
        "content": content or None,
    }

    if refusal:
        message["refusal"] = refusal

    if tool_calls:
        # 함수 호출 요청을 assistant 메시지에 포함
        # 이후 role="tool" 결과 메시지보다 반드시 먼저 기록해야 함
        message["tool_calls"] = tool_calls

    return message


# 사용자 입력 처리 → read_stream() → execute_tool() → 다시 read_stream()
st.title("AI 상담사와 대화하기")

# Streamlit은 입력이나 버튼 클릭 시 스크립트를 다시 실행
# session_state에 저장하면 재실행 사이에도 대화 기록을 유지할 수 있음
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "너는 사용자를 도와주는 상담사야.",
        }
    ]

# 새 대화 버튼을 누르면 현재 세션의 대화 기록을 초기화
if st.sidebar.button("새 대화"):
    st.session_state.messages = [
        {
            "role": "system",
            "content": "너는 사용자를 도와주는 상담사야.",
        }
    ]

    # 초기화된 기록으로 화면을 다시 그림
    st.rerun()

# 이전 대화를 화면에 다시 표시
# system 지시문과 tool 실행 결과는 API용 기록에만 유지
for msg in st.session_state.messages:
    if msg["role"] in ("user", "assistant"):
        text = msg.get("content") or msg.get("refusal")

        if text:
            st.chat_message(msg["role"]).write(text)

# := 는 값을 변수에 저장하면서 조건도 검사하는 문법
# 사용자가 제출한 문자열이 있으면 아래 블록 실행
# 제출하지 않았으면 None이므로 건너뜀
if user_input := st.chat_input("메시지를 입력해 주세요"):
    # 새 사용자 메시지를 화면에 바로 표시
    st.chat_message("user").write(user_input)

    # 기존 기록을 펼치고 새 질문을 덧붙인 임시 리스트 생성
    # 기존 메시지 딕셔너리 자체를 수정하지 않고 새 항목만 추가해서 사용
    #
    # 이번 처리가 성공했을 때만 session_state에 저장하므로
    # 중간 실패로 불완전한 호출 기록이 남는 것을 방지
    pending_messages = [
        *st.session_state.messages,
        {"role": "user", "content": user_input},
    ]

    try:
        # 이번 질문에 대한 AI 출력 영역 생성
        with st.chat_message("assistant"):
            # AI가 함수를 요청하고 결과를 받는 과정을 최대 10회 반복
            # 반복 횟수 제한 없이 함수 호출이 계속 이어지는 상황을 방지
            for _ in range(10):
                # 이번 API 응답을 실시간으로 표시할 빈 영역 생성
                placeholder = st.empty()

                # 스트림을 끝까지 읽고 완성된 assistant 메시지를 받음
                ai_message = read_stream(
                    pending_messages,
                    placeholder,
                )

                # 함수 호출 요청이 담긴 AI 메시지를 먼저 저장
                pending_messages.append(ai_message)

                # tool_calls 키가 없으면 빈 리스트 사용
                tool_calls = ai_message.get("tool_calls", [])

                # 함수 호출이 없으면 최종 답변이 나온 상태
                if not tool_calls:
                    # 성공한 전체 대화 기록을 세션에 반영
                    st.session_state.messages = pending_messages
                    break

                # AI가 요청한 함수를 모두 실행
                for tool_call in tool_calls:
                    name = tool_call["function"]["name"]

                    # 실행 중인 함수 이름을 화면에 표시
                    with st.spinner(f"{name} 실행 중..."):
                        result = execute_tool(tool_call)

                    # 함수 실행 결과를 기록
                    # tool_call_id로 어떤 호출에 대한 결과인지 연결
                    pending_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result,
                        }
                    )

                # 다음 반복에서 함수 결과까지 포함한 기록을 AI에게 전달
                # AI는 최종 답변을 하거나 추가 함수를 요청할 수 있음

            else:
                # 파이썬의 for ... else 문법:
                # break 없이 반복을 모두 마쳤을 때만 실행됨
                raise RuntimeError(
                    "함수 호출 반복 한도에 도달했습니다. "
                    "질문을 나누어 다시 입력해 주세요."
                )

    except Exception as error:
        # API 요청이나 스트림 처리 중 발생한 오류를 화면에 표시
        st.error(f"처리 중 오류가 발생했습니다: {error}")

        # pending_messages를 세션에 반영하지 않았으므로
        # 이번 질문 이전까지의 대화 기록이 유지됨
        st.info("이번 질문은 저장하지 않았어요. 다시 입력해 주세요.")
