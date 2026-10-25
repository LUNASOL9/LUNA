from openai import OpenAI
import pymupdf
import os
from pathlib import Path
from dotenv import load_dotenv

base_dir = "C:\\LUNA\\pdf_summary"
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
def pdf_to_text(pdf_file_path: Path, header_height: int = 80, footer_height: int = 80) -> str:
    
    """
    PDF 파일을 텍스트로 변환하는 함수.
    
    Args:
        pdf_file_path (Path): PDF 파일 경로.
        header_height (int): 헤더 높이 (포인트 단위).
        footer_height (int): 푸터 높이 (포인트 단위).
    
    Returns:
        str: 변환된 텍스트 파일 경로.
    """
    doc = pymupdf.open(pdf_file_path)
    full_text = ""  # 전체 텍스트를 저장할 변수 초기화

    for page in doc:  # 문서 페이지 반복
        rect = page.rect  # 페이지의 사각형 영역 가져오기
        header = page.get_text(clip=(0, 0, rect.width, header_height))  # 헤더 영역에서 텍스트 추출
        footer = page.get_text(clip=(0, rect.height - footer_height, rect.width, rect.height))  # 푸터 영역에서 텍스트 추출

        text = page.get_text(clip=(0, header_height, rect.width, rect.height - footer_height))  # 헤더와 푸터를 제외한 영역에서 텍스트 추출
        full_text += text + "\n--------------------\n"  # 추출한 텍스트를 전체 텍스트 변수에 추가

    pdf_file_name = os.path.basename(pdf_file_path)  # PDF 파일 이름 추출
    pdf_file_name = os.path.splitext(pdf_file_name)[0]  # 확장자 제거

    txt_file_path = Path(base_dir + "\\output") / f"{pdf_file_name}_with_preprocessing.txt"  # TXT 파일 경로 설정

    with open(txt_file_path , "w", encoding="utf-8") as txt_file:  # TXT 파일 열기
        txt_file.write(full_text)

    return txt_file_path




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




def summarize_pdf(pdf_file_path: Path, output_file_path: Path, header_height: int = 50, footer_height: int = 50):
    """
    PDF 파일을 텍스트로 변환하고 요약하는 함수.
    
    Args:
        pdf_file_path (Path): PDF 파일 경로.
        output_file_path (Path): 요약 결과를 저장할 파일 경로.
        header_height (int): 헤더 높이 (포인트 단위).
        footer_height (int): 푸터 높이 (포인트 단위).
    
    Returns:
        str: 요약된 텍스트.
    """
    txt_file_path = pdf_to_text(pdf_file_path, header_height, footer_height)
    summary = summarize_text(txt_file_path)

    with open(output_file_path, "w", encoding="utf-8") as summary_file:  # 요약 결과를 summary.txt 파일에 저장
        summary_file.write(summary)




if __name__ == "__main__":  # 이 파일이 직접 실행될 때만 아래 코드가 실행되도록 함
    pdf_file_path = Path(base_dir + "\\data\\달 자원 탐사와 달 기지 연구 동향.pdf")
    
    output_file_path = Path(base_dir + "\\output\\summary.txt")
    summarize_pdf(pdf_file_path, output_file_path)