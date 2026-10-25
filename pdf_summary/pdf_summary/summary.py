from openai import OpenAI
from dotenv import load_dotenv

import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_dir = "C:\\LUNA\\pdf_summary"

# 텍스트 요약을 수행하는 함수
def summarize_text(file_path: str):
    client = OpenAI(api_key=api_key)

    with open(file_path, "r", encoding="utf-8") as file: # 인자로 받은 파일 경로를 열고, UTF-8 인코딩으로 읽기 모드로 설정
        txt = file.read()


    system_prompt = f'''
    너는 다음 글을 요약하는 봇이다. 아래 글을 읽고, 저자의 문제 인식과 주장을 파악하고, 주요 내용을 요약하라. 

    작성해야 하는 포맷은 다음과 같다. 
    
    # 제목

    ## 저자의 문제 인식 및 주장 (15문장 이내)
    
    ## 저자 소개

    
    =============== 이하 텍스트 ===============

    { txt }
    '''


    print(system_prompt)
    print('=========================================')

    response = client.chat.completions.create(
        model="gpt-6-luna",
        messages=[
            {"role": "system", "content": system_prompt},
        ]    
    )

    return response.choices[0].message.content

if __name__ == "__main__": # 이 파일이 직접 실행될 때만 아래 코드가 실행되도록 함
    file_path = os.path.join(base_dir, "output", "과정기반 작물모형을 이용한 웹 기반 밀 재배관리 의사결정 지원시스템 설계 및 구축_widh_preprocessing.txt.txt")

    summary = summarize_text(file_path) # summarize_text 함수를 호출하여 요약 결과를 summary 변수에 저장
    print(f"\n{summary}")

    with open(os.path.join(base_dir, "output", "summary.txt"), "w", encoding="utf-8") as summary_file: # 요약 결과를 summary.txt 파일에 저장
        summary_file.write(summary)

