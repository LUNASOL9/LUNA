import pymupdf
import os
from pathlib import Path

base_dir = "C:\\LUNA\\pdf_summary"
pdf_file_path = Path(base_dir + "\\data\\과정기반 작물모형을 이용한 웹 기반 밀 재배관리 의사결정 지원시스템 설계 및 구축.pdf")
doc = pymupdf.open(pdf_file_path)

header_height = 80  # 헤더 높이 설정 (단위: 포인트)
footer_height = 80  # 푸터 높이 설정 (단위: 포인트)

full_text = ""  # 전체 텍스트를 저장할 변수 초기화

for page in doc: # 문서 페이지 반복
    rect = page.rect  # 페이지의 사각형 영역 가져오기
    header = page.get_text(clip=(0, 0, rect.width, header_height)) # 헤더 영역에서 텍스트 추출
    footer = page.get_text(clip=(0, rect.height - footer_height, rect.width, rect.height)) # 푸터 영역에서 텍스트 추출

    text = page.get_text(clip=(0, header_height, rect.width, rect.height - footer_height))  # 헤더와 푸터를 제외한 영역에서 텍스트 추출
    full_text += text + "\n--------------------\n" # 추출한 텍스트를 전체 텍스트 변수에 추가

pdf_file_name = os.path.basename(pdf_file_path)  # PDF 파일 이름 추출
pdf_file_name = os.path.splitext(pdf_file_name)[0]  # 확장자 제거

output_dir = Path(base_dir + "\\output")
output_dir.mkdir(parents=True, exist_ok=True)
txt_file_path = output_dir / f"{pdf_file_name}_widh_preprocessing.txt.txt"  # TXT 파일 경로 설정

with open(txt_file_path, "w", encoding="utf-8") as txt_file:  # TXT 파일 열기
    txt_file.write(full_text)  # 추출한 텍스트를 TXT 파일에 쓰기