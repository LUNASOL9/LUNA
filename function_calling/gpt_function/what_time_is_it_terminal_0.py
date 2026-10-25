import os

from dotenv import load_dotenv
from gpt_function import get_current_time, tools
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


# 1 5 1
def get_ai_response(messages, tools=None):
    response = client.chat.completions.create(
        model="gpt-6-luna",
        messages=messages,  # 대화 기록을 입력으로 전달
        tools=tools,  # 사용 가능한 도구 목록 전달
        reasoning_effort="none",  # gpt-6-luna에서 Chat Completions로 함수 도구를 쓰려면 reasoning_effort="none"을 명시해야 해.
    )
    return response


messages = [{"role": "system", "content": "너는 사용자를 도와주는 상담사야."}]

# 사용자가 종료할 때까지 대화 반복
while True:
    user_input = input("User: ")

    # 종료 명령을 입력하면 프로그램 종료
    if user_input.lower() in ["exit", "quit"]:
        break

    # 사용자 질문을 대화 기록에 추가
    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # AI가 함수 실행을 요청하면 처리하고, 최종 답변이 나올 때까지 반복
    while True:
        # 현재 대화 기록과 사용 가능한 도구 목록을 AI에 전달
        ai_response = get_ai_response(messages, tools=tools)

        # 응답에서 AI 메시지를 꺼냄
        ai_message = ai_response.choices[0].message
        print(ai_message)
        # AI 메시지를 딕셔너리로 변환해 대화 기록에 추가
        # 함수 호출 요청(tool_calls)도 여기에 포함됨
        # 반드시 함수 실행 결과보다 먼저 기록해야 함
        messages.append(ai_message.model_dump(exclude_none=True))

        # 함수 호출 요청이 없으면 사용자에게 보여줄 최종 답변
        if not ai_message.tool_calls:
            print("AI:", ai_message.content or "")

            # 안쪽 반복문을 끝내고 다음 사용자 입력을 기다림
            break

        # AI가 요청한 함수를 하나씩 실행
        for tool_call in ai_message.tool_calls:
            # 실행할 함수 이름
            tool_name = tool_call.function.name

            if tool_name == "get_current_time":
                # 함수가 반환한 현재 시간을 저장
                result = get_current_time()
            else:
                # 등록되지 않은 함수는 실행하지 않고 안내 문구를 반환
                result = f"지원하지 않는 함수: {tool_name}"

            # 함수 실행 결과를 대화 기록에 추가
            messages.append(
                {
                    "role": "tool",
                    # 어떤 함수 호출에 대한 결과인지 연결하는 ID
                    "tool_call_id": tool_call.id,
                    # 함수 결과는 문자열로 전달
                    "content": str(result),
                }
            )

        # 안쪽 반복문 처음으로 돌아가 함수 결과를 AI에 전달
        # AI는 결과를 참고해 답변하거나 추가 함수를 요청할 수 있음
