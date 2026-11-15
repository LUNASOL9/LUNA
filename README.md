# LUNA · LLM 활용 실습 모음

OpenAI API와 Whisper로 **챗봇 → 프롬프트 → Function Calling → 문서 요약 → 회의록 자동화**까지 단계별로 직접 만들어 본 Python 실습 저장소입니다.

각 폴더가 하나의 실습이며, 앞 단계에서 익힌 내용을 다음 단계에서 다시 사용하는 순서로 구성했습니다.

<br>

## 실습 한눈에 보기

| 단계 | 폴더 | 내용 | 핵심 기술 |
|:---:|---|---|---|
| 1 | [`gpt_basic`](gpt_basic) | 첫 Chat Completions API 호출 | `openai`, `python-dotenv` |
| 2 | [`gpt_chatbot/multi_turn`](gpt_chatbot/multi_turn) | 싱글턴 vs 멀티턴 챗봇 비교 | 대화 기록(`messages`) 관리 |
| 3 | [`gpt_chatbot/prompting`](gpt_chatbot/prompting) | Zero / One / Few-shot, 역할 부여 | 프롬프트 엔지니어링 |
| 4 | [`function_calling`](function_calling) | GPT가 함수를 골라 호출하는 챗봇 (현재 시각, 주식 정보) · Streamlit 웹 화면 · 스트리밍 응답 | `tools`, `tool_calls`, `yfinance`, Streamlit |
| 5 | [`pdf_summary`](pdf_summary) | 논문 PDF → 텍스트 추출 → 구조화된 요약 | `PyMuPDF` |
| 6 | [`whisper`](whisper) | 음성 인식 + 화자 분리 + 회의록 자동 작성 | Whisper, `pyannote.audio`, `pandas`, `python-docx` |

<br>

## 기술 스택

| 분류 | 사용 기술 |
|---|---|
| 언어 / 환경 | Python 3.12, venv, Jupyter Notebook, VS Code + Ruff |
| LLM | OpenAI Chat Completions (`gpt-6-luna`) |
| 웹 UI | Streamlit |
| 음성 인식 | OpenAI Whisper API (`whisper-1`), Hugging Face Transformers (`openai/whisper-large-v3-turbo`) |
| 화자 분리 | `pyannote.audio` 3.3 (`pyannote/speaker-diarization-3.1`) |
| 딥러닝 | PyTorch 2.7 (CUDA 12.8), torchaudio, FFmpeg |
| 데이터 처리 | yfinance, PyMuPDF, pandas, pytz, tabulate, python-docx |

<br>

## 폴더 구조

```text
LUNA/
├── gpt_basic/
│   └── gpt_basic.py                     # 첫 API 호출
├── gpt_chatbot/
│   ├── multi_turn/
│   │   ├── single_turn.py               # 이전 대화를 기억하지 않는 챗봇
│   │   └── multi_turn.py                # 대화 기록을 유지하는 챗봇
│   └── prompting/
│       ├── few_shot/                    # no_prompting · one_shot · few_shot
│       └── mirror_in_snow_white/        # 역할 부여 프롬프트
├── function_calling/
│   ├── gpt_function/
│   │   ├── gpt_function.py              # get_current_time() + tools 스키마
│   │   ├── what_time_is_it_terminal_0.py  # 터미널 챗봇
│   │   └── what_time_is_it_streamlit.py   # Streamlit 웹 챗봇
│   ├── stock_info/
│   │   ├── yfinance.ipynb               # yfinance 사용법 연습
│   │   ├── gpt_function.py              # 시각 + 주식 정보 함수 4개 + tools 스키마
│   │   └── stock_info_streamlit.py      # 주식 정보 웹 챗봇
│   └── stock_info_streaming/
│       ├── gpt_function.py              # stock_info와 같은 함수 파일
│       └── stock_info_streamlit.py      # 답변을 실시간으로 보여 주는 스트리밍 버전
├── pdf_summary/
│   ├── data/                            # 원본 논문 PDF
│   ├── output/                          # 추출한 텍스트 · 요약 결과
│   ├── pdf_without_header_footer/       # PDF → TXT 변환
│   └── pdf_summary/                     # TXT → GPT 요약
├── whisper/
│   ├── audio/                           # 원본 음성 · STT/화자 분리 결과
│   ├── whispe_api/                      # OpenAI Whisper API
│   ├── huggingface_whisper/             # 로컬 Whisper 모델
│   ├── speaker_diarization/             # STT + 화자 분리 + 회의록 자동 작성 스크립트
│   └── summarize_and_correct/           # 회의록 요약 · STT 교정 · 문서 내보내기 (노트북)
├── .vscode/
│   ├── settings.json                    # Ruff 자동 정렬, .env 자동 로드
│   └── launch.json                      # F5로 Streamlit 디버깅
├── .env                                 # API 키 (Git에서 제외)
├── SETUP.md                             # Windows 설치 가이드
└── README.md
```

<br>

## 시작하기

> Windows에서 처음부터 설치하는 자세한 과정과 문제 해결 방법은 [SETUP.md](SETUP.md)에 정리했습니다.

### 1. 준비물

- Python 3.12
- OpenAI API 키
- 6단계(whisper)를 실행하려면 추가로
  - Hugging Face 토큰
  - Hugging Face에서 [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1), [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) 모델 이용 약관 동의
  - FFmpeg
  - NVIDIA GPU 권장 (없으면 CPU로 동작하지만 느립니다)

### 2. 가상환경 만들기

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. 패키지 설치

```powershell
# 1~5단계
pip install openai python-dotenv pytz pymupdf streamlit yfinance tabulate ipykernel

# 6단계 (whisper)
pip install "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" "torchaudio==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
pip install "pyannote.audio==3.3.2" "transformers==4.55.4" "huggingface-hub==0.34.4" "accelerate==1.15.0" "numpy==2.5.4" pandas python-docx
pip install "pytorch-metric-learning<3" "torch==2.7.1+cu128" --extra-index-url https://download.pytorch.org/whl/cu128
```

> [!WARNING]
> 6단계의 세 번째 줄을 빼먹으면 `pytorch-metric-learning` 3.x가 설치되면서 GPU용 torch가 CPU용 torch로 바뀝니다. 자세한 내용은 [SETUP.md](SETUP.md)를 참고하세요.

### 4. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 만들고 키를 넣습니다. `.env`는 `.gitignore`에 등록되어 있어 커밋되지 않습니다.

```env
OPENAI_API_KEY=sk-...
HUGGING_FACE_TOKEN=hf_...
```

### 5. FFmpeg

음성 파일을 읽을 때 FFmpeg가 필요합니다. 둘 중 하나로 준비합니다.

- `winget install --id Gyan.FFmpeg -e --source winget`으로 설치해 시스템 `PATH`에 등록
- [FFmpeg 빌드](https://www.gyan.dev/ffmpeg/builds/)를 프로젝트 루트에 `ffmpeg-2026-10-15-git-afead44165-full_build` 폴더로 풀기 (코드가 실행될 때 `bin` 폴더를 `PATH`에 자동으로 추가하며, 이 폴더는 `.gitignore` 처리되어 있습니다)

> [!NOTE]
> 코드 안의 파일 경로는 `C:\LUNA` 기준으로 적혀 있습니다.
> 다른 위치에 clone했다면 각 스크립트의 `base_dir`, `AUDIO`, `FFMPEG_BIN` 같은 경로 변수를 바꿔 주세요.

<br>

## 실습 상세

### 1. gpt_basic — 첫 API 호출

`.env`에서 API 키를 읽어 `system`, `user` 메시지 한 쌍을 보내고 답변을 출력합니다.
이후 모든 실습이 이 구조(`load_dotenv` → `OpenAI` 클라이언트 → `chat.completions.create`)를 기본으로 사용합니다.

```powershell
python gpt_basic\gpt_basic.py
```

<br>

### 2. gpt_chatbot/multi_turn — 싱글턴 vs 멀티턴

| 파일 | 동작 | 결과 |
|---|---|---|
| `single_turn.py` | 매번 `system` + 현재 질문만 전송 | 바로 앞의 대화도 기억하지 못함 |
| `multi_turn.py` | `messages` 리스트에 `user`/`assistant` 메시지를 계속 추가해 전송 | 이전 대화를 이어서 답변 |

API 서버는 대화를 저장하지 않습니다. 챗봇의 "기억"은 **클라이언트가 지금까지의 대화 기록을 매번 다시 보내서** 만들어진다는 점을 두 파일로 비교했습니다.

```powershell
python gpt_chatbot\multi_turn\multi_turn.py   # exit 또는 quit 입력 시 종료
```

<br>

### 3. gpt_chatbot/prompting — 프롬프트 엔지니어링

"너는 유치원생이야"라는 같은 `system` 프롬프트에서 **예시 개수**만 바꿔 답변이 어떻게 달라지는지 비교합니다.

| 파일 | 방식 | 대화 예시 → 마지막 질문 |
|---|---|---|
| `few_shot/no_prompting.py` | Zero-shot | (예시 없음) → 오리 |
| `few_shot/one_shot.py` | One-shot | 참새 → 짹짹 → 오리 |
| `few_shot/few_shot.py` | Few-shot | 참새 → 짹짹, 말 → 히이잉, 개구리 → 개굴개굴 → 뱀 |
| `mirror_in_snow_white/mirror_in_snow_white.py` | 역할 부여 | 백설공주 속 "마법 거울" 캐릭터로 답변 |

`user`/`assistant` 메시지 쌍으로 예시를 넣으면, 모델이 그 형식(울음소리로 답하기)을 따라 답변하게 됩니다.

<br>

### 4. function_calling — 함수를 호출하는 챗봇

GPT는 현재 시각이나 오늘의 주가를 알 수 없습니다. 그래서 이런 정보를 가져오는 파이썬 함수를 **도구(tool)** 로 등록하고, 필요할 때 GPT가 그 함수를 호출하도록 만들었습니다.
터미널 챗봇 → Streamlit 웹 챗봇 → 주식 정보 도구 추가 → 스트리밍 응답 순서로 발전시켰습니다.

```mermaid
sequenceDiagram
    participant U as 사용자
    participant A as 챗봇 (Python)
    participant G as GPT
    participant F as get_current_time()

    U->>A: 뉴욕은 지금 몇 시야?
    A->>G: 대화 기록 + tools
    G-->>A: tool_calls 요청 (timezone = America/New_York)
    A->>F: 함수 실행
    F-->>A: 현재 날짜와 시각 (America/New_York)
    A->>G: role tool 메시지로 결과 전달
    G-->>A: 최종 답변
    A->>U: 뉴욕의 현재 시각 안내
```

#### 4-1. 현재 시각 챗봇 — `gpt_function/`

| 파일 | 역할 |
|---|---|
| `gpt_function.py` | `get_current_time(timezone)` 함수와, GPT에게 함수를 설명하는 `tools` JSON 스키마 |
| `what_time_is_it_terminal_0.py` | Function Calling 흐름을 처리하는 터미널 챗봇 |
| `what_time_is_it_streamlit.py` | 같은 흐름을 Streamlit 채팅 화면으로 옮긴 버전 |

**구현 포인트**

- GPT가 함수 호출을 요청하지 않을 때까지 안쪽 `while` 루프를 반복해, 한 질문에 여러 번 함수를 호출해도 처리할 수 있습니다.
- `tool_calls`가 담긴 `assistant` 메시지를 **함수 결과보다 먼저** 대화 기록에 추가해야 합니다.
- 함수 결과는 `role: "tool"` 메시지에 `tool_call_id`를 붙여, 어떤 호출에 대한 결과인지 연결합니다.
- GPT가 넘겨주는 `arguments`는 JSON 문자열이라 `json.loads()`로 파싱합니다.
- 등록되지 않은 함수 이름이 오면 실행하지 않고 안내 문구를 돌려줍니다.
- `gpt-6-luna`에서 Chat Completions로 함수 도구를 쓰려면 `reasoning_effort="none"`을 지정해야 합니다.
- Streamlit은 입력할 때마다 스크립트를 처음부터 다시 실행하므로, 대화 기록을 `st.session_state`에 저장해 유지합니다. 화면에는 `user`/`assistant` 메시지만 보여 주고 `system`, `tool` 메시지는 숨깁니다.

```powershell
cd function_calling\gpt_function   # gpt_function 모듈을 같은 폴더에서 import합니다
python what_time_is_it_terminal_0.py
```

```powershell
streamlit run function_calling\gpt_function\what_time_is_it_streamlit.py
```

#### 4-2. 주식 정보 챗봇 — `stock_info/`

먼저 `yfinance.ipynb`에서 Microsoft(`MSFT`)를 예로 기업 정보, 최근 주가, 애널리스트 추천을 가져오는 방법을 연습한 뒤, 이를 함수로 만들어 도구로 등록했습니다.

| 함수 | 하는 일 |
|---|---|
| `get_current_time(timezone)` | 지정한 타임존의 현재 날짜와 시각 |
| `get_yf_stock_info(ticker)` | 회사 개요, 현재 주가, 재무 정보 등 |
| `get_yf_stock_history(ticker, period)` | 기간별 과거 주가 (`1d`, `5d`, `1mo`, `1y`, `max` 등) |
| `get_yf_stock_recommendations(ticker)` | 애널리스트 매수·매도 추천 |

- 주가 데이터는 pandas DataFrame이라서, `to_markdown()`(`tabulate` 패키지 필요)으로 표 형태의 문자열로 바꿔 GPT에게 넘깁니다.
- 함수 실행 중 오류가 나도 앱이 멈추지 않도록 `try/except`로 감싸고, 오류 내용을 `tool` 메시지로 GPT에게 전달합니다.

```powershell
streamlit run function_calling\stock_info\stock_info_streamlit.py
```

> 주식 정보 챗봇은 학습용 예제이며, 투자 판단의 근거로 쓰기 위한 것이 아닙니다.

#### 4-3. 스트리밍 버전 — `stock_info_streaming/`

`stream=True`로 요청해 답변이 만들어지는 대로 화면에 한 글자씩 보여 줍니다.
스트리밍에서는 함수 호출 정보도 여러 조각으로 나뉘어 오기 때문에, 조각을 모두 모은 뒤에 함수를 실행합니다.

```mermaid
flowchart LR
    A["스트림 청크"] --> B{"delta 종류"}
    B -->|content| C["화면에 바로 이어 쓰기"]
    B -->|tool_calls 조각| D["index별로 id · name ·<br/>arguments 이어 붙이기"]
    D --> E["스트림이 끝나면<br/>함수 실행"]
    E --> F["tool 메시지 추가 후<br/>다시 요청"]
    F --> A
```

**구현 포인트**

- `st.empty()`로 만든 자리를 계속 갱신해, 답변이 한 위치에서 길어지는 것처럼 보이게 합니다.
- `FUNCTIONS` 딕셔너리에 함수 이름과 실제 함수를 짝지어 두고, `FUNCTIONS[name](**arguments)`로 실행합니다. `if/elif`를 늘리지 않아도 함수를 추가할 수 있습니다.
- `finish_reason`이 `stop`이나 `tool_calls`가 아니면 응답이 중간에 끊긴 것으로 보고 실행하지 않습니다.
- 함수 호출 반복은 최대 10번으로 제한합니다. (`for ... else` 문법 사용)
- 대화 기록은 임시 리스트(`pending_messages`)에 쌓다가 끝까지 성공했을 때만 `st.session_state`에 저장해, 중간에 실패해도 불완전한 기록이 남지 않습니다.
- 사이드바의 **새 대화** 버튼으로 대화 기록을 초기화합니다.

```powershell
streamlit run function_calling\stock_info_streaming\stock_info_streamlit.py
```

> VS Code에서는 Streamlit 파일을 연 상태로 **F5**(실행 및 디버그 → `Streamlit 디버깅`)를 누르면 중단점을 걸고 디버깅할 수 있습니다. (`.vscode/launch.json`)

<br>

### 5. pdf_summary — 논문 PDF 요약

```mermaid
flowchart LR
    A["논문 PDF"] --> B["머리글·바닥글 제거<br/>텍스트 추출"]
    B --> C["TXT 파일"]
    C --> D["GPT 요약"]
    D --> E["summary.txt"]
```

| 파일 | 역할 |
|---|---|
| `pdf_without_header_footer/pdf_to_txt.py` | PDF 전체 텍스트를 그대로 추출 |
| `pdf_without_header_footer/pdf_without_header_footer.py` | 페이지 위·아래 영역을 잘라내고(`clip`) 본문만 추출 |
| `pdf_summary/summary.py` | 추출한 TXT를 GPT로 요약 |
| `pdf_summary/pdf_summary.py` | 추출과 요약을 `summarize_pdf()` 함수 하나로 묶은 최종본 |

- 학술지 이름, 쪽 번호처럼 페이지마다 반복되는 머리글·바닥글을 PyMuPDF의 `clip` 영역 지정으로 제외해 요약에 불필요한 텍스트가 섞이지 않게 했습니다.
- 요약 형식은 프롬프트로 고정했습니다: `# 제목` / `## 저자의 문제 인식 및 주장 (15문장 이내)` / `## 저자 소개`
- 결과 예시: [`pdf_summary/output/summary.txt`](pdf_summary/output/summary.txt) — 「달 자원 탐사와 달 기지 연구 동향」 요약

```powershell
python pdf_summary\pdf_summary\pdf_summary.py
```

<br>

### 6. whisper — 음성 인식 · 화자 분리 · 회의록 자동화

음성 파일에서 **누가, 언제, 무엇을 말했는지** 뽑아내고, GPT로 회의록을 요약·교정해 문서로 내보냅니다.
마지막 단계의 `meeting_minutes.py`는 이 과정 전체를 명령 한 번으로 실행합니다.

```mermaid
flowchart LR
    A["음성 파일<br/>(mp3 · m4a)"] --> B["Whisper STT<br/>단어별 타임스탬프"]
    A --> C["pyannote<br/>화자 분리"]
    B --> D["겹치는 시간으로<br/>단어마다 화자 배정"]
    C --> D
    D --> E["화자 ID →<br/>실제 이름"]
    E --> F["GPT 회의 요약"]
    E --> G["GPT STT 교정"]
    F --> H["summary.md"]
    G --> I["Markdown · JSON · Word 회의록"]
```

#### 6-1. 음성 인식 두 가지 방법

| 노트북 | 방식 | 특징 |
|---|---|---|
| `whispe_api/openai_api_whisper.ipynb` | OpenAI Whisper API (`whisper-1`) | 설치 없이 API로 바로 변환 |
| `huggingface_whisper/huggingface_whisper.ipynb` | 로컬 모델 (`openai/whisper-large-v3-turbo`) | GPU에서 float16으로 실행, 10초 단위로 나누고 2초씩 겹쳐 인식, 구간별 타임스탬프를 CSV로 저장 |

#### 6-2. 화자 분리 — 버전별 개선 과정

`speaker_diarization/` 폴더에는 문제를 하나씩 해결하며 고쳐 온 스크립트가 순서대로 남아 있습니다.

| 파일 | 내용 |
|---|---|
| `whisper_stt_0.py` | Whisper STT 결과를 `start`, `end`, `text` CSV로 저장 |
| `whisper_stt_1.py` | pyannote 3.1 화자 분리 추가. RTTM 파일을 CSV로 바꾸고, 같은 화자가 이어 말한 구간을 하나로 묶음 |
| `whisper_stt_fail.py` | STT 문장 청크와 화자 구간을 겹치는 시간으로 매칭한 첫 시도. 청크 단위라 한 청크에 여러 화자가 섞이면 한 사람에게 몰리고, 겹치는 구간이 없는 청크는 결과에서 빠짐 |
| `whisper_stt_last.py` | **최종본.** 아래 개선 사항 반영 |

`whisper_stt_last.py`에서 개선한 점

- **단어 단위 타임스탬프**(`return_timestamps="word"`)로 화자를 배정해 정확도를 높였습니다.
- 인식된 단어를 하나도 버리지 않고, 화자를 정할 수 없으면 `UNKNOWN`으로 남깁니다.
- 단어마다 배정 상태를 기록하고, 확인이 필요한 항목만 `review.csv`로 따로 모읍니다.

  | status | 의미 |
  |---|---|
  | `matched` | 한 화자 구간과 50% 이상 겹침 |
  | `partial_overlap` | 겹치지만 50% 미만 |
  | `multiple_speakers` | 여러 화자 구간과 겹침 |
  | `no_overlap` | 겹치는 화자 구간이 없음 → `UNKNOWN` |
  | `invalid_timestamp` | 타임스탬프가 없거나 비정상 → `UNKNOWN` |

- 실행할 때마다 `오디오이름_result_날짜시간/` 폴더를 새로 만들어 이전 결과를 보존합니다.
- STT가 끝나면 Whisper 모델을 메모리에서 해제한 뒤 화자 분리를 실행해 GPU 메모리 부족을 막습니다.
- CSV는 `utf-8-sig`로 저장해 엑셀에서 열어도 한글이 깨지지 않습니다.

생성되는 파일

| 파일 | 내용 |
|---|---|
| `stt_raw.json` | Whisper 원본 결과 |
| `speakers.rttm`, `speaker_segments.csv` | 화자 분리 결과 |
| `words.csv` | 단어별 화자 배정 결과 |
| `review.csv` | 검토가 필요한 단어 목록 |
| `*_by_Whisper_stt_Final.csv` | 화자 구간별로 텍스트를 합친 최종 회의록 |

스크립트 맨 위의 상수로 설정을 바꿀 수 있습니다.

```python
NUM_SPEAKERS = 4      # 화자 수 (모르면 None)
LANGUAGE = "korean"   # 자동 감지는 None
MODEL_ID = "openai/whisper-large-v3-turbo"
```

```powershell
python whisper\speaker_diarization\whisper_stt_last.py
```

#### 6-3. 회의록 요약 · 교정 · 내보내기 (노트북)

`summarize_and_correct/summarize_and_correct.ipynb`

1. 최종 회의록 CSV를 불러와 빈 발화와, 분석에서 뺄 화자(`SPEAKER_00`)·`UNKNOWN`을 제외합니다.
2. `SPEAKER_01` 같은 화자 ID를 실제 인물 이름으로 바꿉니다.
3. 회의록을 JSON으로 바꿔 GPT에 요약을 요청합니다. (`# 회의 제목` / `## 주요 내용` / `## 참석자별 입장` / `## 결정 사항`) → [`summary.md`](whisper/summarize_and_correct/summary.md)
4. 발화를 하나씩 GPT로 교정합니다. 회의 요약과 전체 회의록을 함께 넘겨, 앞뒤 맥락을 보고 STT가 잘못 받아 적은 부분만 고치게 했습니다.
5. 교정한 회의록을 JSON, Markdown, Word(`.docx`) 파일로 저장합니다.

#### 6-4. 회의록 자동 작성 — `meeting_minutes.py`

6-2의 STT·화자 분리와 6-3의 요약·교정을 하나로 이은 스크립트입니다. 음성 파일 이름, 화자 수, 화자 이름만 입력하면 회의록 문서까지 만들어 줍니다.

```powershell
python whisper\speaker_diarization\meeting_minutes.py
```

```text
음성 파일 이름.확장자 (예: meeting_record.m4a): meeting_record.m4a
화자 수: 3
화자 이름 (쉼표로 구분, 최대 3명): 루피, 도라, 얀
```

| 단계 | 내용 |
|---|---|
| 1. 오디오 읽기 | FFmpeg로 16kHz 모노 파형으로 변환합니다. mp3, m4a, wav, mp4 등 FFmpeg가 읽는 형식은 모두 됩니다. |
| 2. Whisper STT | 단어별 타임스탬프로 음성을 인식합니다. |
| 3. 화자 분리 | 입력한 화자 수로 구간을 나누고, `whisper_stt_last.py`의 함수로 단어마다 화자를 배정합니다. |
| 4. 화자 이름 정하기 | 화자마다 가장 긴 발화 3개를 보여 주고, 누구인지 번호로 고릅니다. |
| 5. GPT 회의 요약 | 6-3과 같은 형식으로 요약합니다. |
| 6. GPT STT 교정 | 발화를 20개씩 묶어 교정하고, 결과를 JSON 형식으로 받습니다. |

**구현 포인트**

- 화자 이름의 기본값은 "아직 고르지 않은 이름 중 맨 앞 이름"입니다. 처음 말한 순서대로 이름을 입력했다면 Enter만 눌러도 됩니다. 배경음처럼 뺄 화자는 `0`을 고릅니다.
- 같은 사람이 이어서 말한 발화는 하나로 합칩니다.
- 노트북처럼 발화마다 전체 회의록을 보내면 긴 회의에서 호출 수와 비용이 크게 늘어납니다. 그래서 20개씩 묶어 보내고, 회의 요약과 앞뒤 5개 발화만 참고용으로 함께 보냅니다.
- `response_format={"type": "json_object"}`로 JSON 답변을 받고, 답을 읽지 못하거나 빠진 발화는 원문을 그대로 씁니다.
- `whisper_stt_last.py`의 `assign_words`, `build_final`, `save_csv`를 import해서 쓰므로 두 파일이 같은 폴더에 있어야 합니다.

결과는 `whisper/audio/<파일이름>_minutes_<날짜시간>/` 폴더에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `summary.md` | 회의 요약 |
| `meeting_minutes.md` | 요약 + 시간·이름이 붙은 녹취록 |
| `meeting_minutes.docx` | 요약 + 녹취록 표 (시간 / 화자 / 내용) |
| `meeting_minutes.json` | 발화별 원문과 교정본 |
| `words.csv`, `review.csv`, `*_by_Whisper_stt_Final.csv` 등 | 6-2와 같은 중간 결과 |

<br>

## 배운 점 · 주의할 점

| 상황 | 방법 |
|---|---|
| 챗봇이 이전 대화를 기억하게 하려면 | API는 상태를 저장하지 않으므로 `messages`에 대화를 누적해 매번 전송 |
| Function Calling에서 함수 결과를 GPT에 넘길 때 | `tool_calls`가 담긴 `assistant` 메시지를 먼저 기록한 뒤 `tool` 메시지 추가 |
| `gpt-6-luna`에서 Chat Completions로 함수 도구를 쓸 때 | `reasoning_effort="none"` 지정 |
| Streamlit에서 입력할 때마다 대화가 사라짐 | 스크립트가 매번 다시 실행되므로 `st.session_state`에 대화 기록 저장 |
| 스트리밍 중 함수 인자가 여러 조각으로 옴 | `index`별로 문자열을 이어 붙였다가, 스트림이 끝난 뒤 `json.loads()` |
| DataFrame을 GPT에게 넘길 때 | `to_markdown()`으로 표 문자열로 변환 (`tabulate` 필요) |
| PDF 요약에 쪽 번호·학술지명이 섞임 | PyMuPDF `clip`으로 머리글·바닥글 영역 제외 |
| Whisper가 mp3를 읽지 못함 | FFmpeg `bin` 폴더를 코드에서 `PATH`에 추가 |
| torchaudio가 m4a를 읽지 못함 | soundfile 백엔드는 m4a를 지원하지 않아서, `ffmpeg.exe`로 16kHz 모노 PCM으로 직접 변환 |
| pyannote 설치 후 GPU를 못 씀 | `pytorch-metric-learning<3`으로 고정해 CPU용 torch로 바뀌는 것을 막음 |
| pyannote 모델 로딩 실패 | PyTorch 2.6부터 `torch.load` 기본값이 `weights_only=True`로 바뀌어, `torch.serialization.safe_globals`로 허용할 클래스를 지정 |
| 화자 분리 중 GPU 메모리 부족 | STT 후 모델 삭제 + `torch.cuda.empty_cache()` |
| 한 문장에 여러 화자가 섞여 배정이 틀림 | 문장 청크 대신 단어 단위 타임스탬프로 화자 배정 |
| 긴 회의를 발화마다 교정하면 느리고 비쌈 | 20개씩 묶어 교정하고 JSON 형식으로 답 받기 |
| 엑셀에서 CSV 한글이 깨짐 | `utf-8-sig` 인코딩으로 저장 |
