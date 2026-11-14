# LUNA 설치 가이드 (Windows / PowerShell)

LLM 활용 실습 모음(OpenAI API, Function Calling, PDF 요약, Whisper + pyannote 화자 분리) 개발 환경을 이 PC(`C:\LUNA`) 기준으로 정리한 문서입니다.

- 작성일: 2026-11-14
- 대상 PC: Intel i5-14600KF / RAM 32GB / NVIDIA RTX 4060 Ti 8GB (드라이버 610.88) / Windows 11 Pro

## 1. 현재 설치 상태

| 항목 | 요구 버전 | 이 PC 상태 | 설치 방법 |
| --- | --- | --- | --- |
| Git for Windows | 최신 | 2.56.0 설치됨 | `winget install --id Git.Git -e --source winget` |
| Python | 3.12 | 3.12.10 설치됨 | `winget install --id Python.Python.3.12 -e --source winget` |
| 가상환경 | `venv` | `C:\LUNA\venv` 생성됨 | `python -m venv venv` |
| 기본 패키지 (1~5단계) | openai, python-dotenv, pytz, pymupdf | 설치됨 | 3절 |
| 추가 패키지 | streamlit, yfinance, ipykernel | 설치됨 | 3절 (README에는 없지만 코드가 import) |
| GPU 패키지 (6단계) | torch 2.7.1+cu128, pyannote.audio 3.3.2 등 | 설치됨 (CUDA 인식 확인) | 3절 |
| FFmpeg | 빌드 폴더 또는 PATH | 9.0.2 설치됨 (PATH) | `winget install --id Gyan.FFmpeg -e --source winget` |
| VS Code 확장 | Python, Ruff, Jupyter | 설치됨 | `code --install-extension ...` |
| OpenAI API 키 | 필수 | `.env`에 설정됨 (호출 확인) | 2절 |
| Hugging Face 토큰 + 모델 약관 동의 | 6단계에 필수 | `.env`에 설정됨 (모델 접근 확인) | 2절 |

## 2. 직접 해야 할 것

### 2.1 OpenAI API 키

1. <https://platform.openai.com/api-keys> 에 로그인해 "Create new secret key"로 키를 만듭니다. (결제 수단 등록과 크레딧이 필요합니다.)
2. 키는 한 번만 보이므로 바로 복사합니다.

### 2.2 Hugging Face 토큰과 모델 약관 (Whisper 화자 분리 실습용)

1. <https://huggingface.co/join> 에서 가입합니다.
2. 아래 두 모델 페이지에 로그인한 상태로 들어가 약관에 동의합니다. **둘 다** 해야 합니다.
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
3. <https://huggingface.co/settings/tokens> 에서 Read 권한 토큰을 만듭니다.

### 2.3 `.env` 파일 만들기

프로젝트 루트(`C:\LUNA\.env`)에 직접 만듭니다. `.gitignore`에 들어 있어 커밋되지 않습니다. 키를 채팅이나 코드에 붙여 넣지 마세요.

```powershell
notepad C:\LUNA\.env
```

내용:

```
OPENAI_API_KEY=sk-...
HUGGING_FACE_TOKEN=hf_...
```

## 3. 패키지 설치 (이미 완료한 내용)

```powershell
cd C:\LUNA
python -m venv venv
.\venv\Scripts\Activate.ps1
```

`Activate.ps1`이 실행 정책 오류로 막히면 현재 사용자에 한해 허용합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

1~5단계(기본 실습)와 Streamlit/yfinance 예제:

```powershell
pip install openai python-dotenv pytz pymupdf streamlit yfinance ipykernel
```

6단계(Whisper + 화자 분리, GPU):

```powershell
pip install "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" "torchaudio==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
pip install "pyannote.audio==3.3.2" "transformers==4.55.4" "huggingface-hub==0.34.4" "accelerate==1.15.0" "numpy==2.5.4" pandas python-docx
pip install "pytorch-metric-learning<3" "torch==2.7.1+cu128" --extra-index-url https://download.pytorch.org/whl/cu128
```

> **주의 (2026-11-14에 실제로 겪은 문제):** README의 두 줄만 실행하면 두 번째 `pip install`이 `pytorch-metric-learning 3.0.3`(torch 2.14 이상 요구)을 끌어오면서 GPU용 `torch 2.7.1+cu128`을 CPU용 `torch 2.15.0`으로 바꿔 버립니다. 세 번째 줄이 이를 막습니다. 이 PC에는 `pytorch-metric-learning 2.9.0`으로 맞춰 두었고 `pip check`에서 충돌이 없습니다.

> CUDA Toolkit을 따로 설치할 필요는 없습니다. `+cu128` 휠 안에 CUDA 12.8 런타임이 들어 있고, NVIDIA 드라이버만 있으면 됩니다.

GPU 확인:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

이 PC에서 확인한 결과(2026-11-14): `2.7.1+cu128 True NVIDIA GeForce RTX 4060 Ti`. 모든 패키지 import와 `torchaudio.load`로 `Forward_Base_Taking_a_Day_Off.mp3` 읽기도 성공했습니다. `.env` 작성 후 `python gpt_basic\gpt_basic.py`가 응답을 받았고, Hugging Face 토큰으로 `pyannote/speaker-diarization-3.1`과 `pyannote/segmentation-3.0` 두 모델에 접근되는 것도 확인했습니다. `whisper_stt_last.py` 전체 실행(모델 다운로드 포함)은 아직 해 보지 않았습니다.

## 4. FFmpeg

코드는 `C:\LUNA\ffmpeg-2026-10-15-git-afead44165-full_build\bin`을 실행 중에 PATH에 추가하도록 되어 있습니다. 이 PC에는 그 폴더 대신 winget으로 FFmpeg 9.0.2를 설치해 **시스템 PATH에 이미 `ffmpeg`가 있습니다.** `whisper_stt_last.py`는 폴더가 없으면 그냥 건너뛰므로 코드 수정 없이 동작합니다.

```powershell
ffmpeg -version
```

README와 똑같은 폴더 구성을 원하면 <https://www.gyan.dev/ffmpeg/builds/> 에서 full build를 받아 프로젝트 루트에 위 이름의 폴더로 풉니다(선택).

## 5. 실행

가상환경을 켠 상태(`.\venv\Scripts\Activate.ps1`)에서 실행합니다.

| 단계 | 명령 | 필요한 것 |
| --- | --- | --- |
| GPT 기본 | `python gpt_basic\gpt_basic.py` | OpenAI 키 |
| 멀티턴 챗봇 | `python gpt_chatbot\multi_turn\multi_turn.py` (`exit`/`quit`로 종료) | OpenAI 키 |
| Function Calling | `cd function_calling\gpt_function` 후 `python what_time_is_it_terminal_0.py` | OpenAI 키 |
| Function Calling (웹) | `streamlit run function_calling\gpt_function\what_time_is_it_streamlit.py` | OpenAI 키 |
| 주식 정보 (웹) | `streamlit run function_calling\stock_info\stock_info_streamlit.py` | OpenAI 키 |
| PDF 요약 | `python pdf_summary\pdf_summary\pdf_summary.py` | OpenAI 키 |
| Whisper + 화자 분리 | `python whisper\speaker_diarization\whisper_stt_last.py` | HF 토큰, 약관 동의, GPU |

노트북(`.ipynb`)은 VS Code에서 열고 오른쪽 위 커널 선택에서 `venv (Python 3.12.10)`을 고릅니다.

### 8GB VRAM 참고

`openai/whisper-large-v3-turbo`(약 1.6GB, 첫 실행 때 자동 다운로드)와 `pyannote/speaker-diarization-3.1`은 RTX 4060 Ti 8GB에서 함께 돌릴 수 있습니다. 다만 Ollama가 `qwen3:8b`를 올려 둔 상태(VRAM 약 6GB 사용)에서는 메모리가 모자랄 수 있으니, Whisper 실습 전에 모델을 내립니다.

```powershell
ollama stop qwen3:8b
nvidia-smi
```

## 6. 경로 주의

스크립트는 `C:\LUNA` 기준 절대 경로를 씁니다(`base_dir`, `AUDIO`, `FFMPEG_BIN`). 이 PC는 `C:\LUNA`에 있으므로 수정할 필요가 없습니다.

## 7. 문제 해결

| 증상 | 확인할 것 |
| --- | --- |
| `python`이 Microsoft Store를 엶 | 설정 → 앱 → 고급 앱 설정 → 앱 실행 별칭에서 `python.exe`, `python3.exe` 끄기. 또는 `py -3.12` 사용 |
| `OPENAI_API_KEY` 관련 오류 | `C:\LUNA\.env` 위치와 변수 이름 확인, 터미널을 `C:\LUNA`에서 실행 |
| pyannote 모델 401/403 | 2.2의 두 모델 약관 동의와 토큰 확인 |
| `torch.cuda.is_available()`이 False | `pip list`에서 torch가 `+cu128`인지 확인. `+cpu`면 3절 명령으로 재설치 |
| CUDA out of memory | `ollama stop qwen3:8b`로 VRAM 확보 |
| mp3/m4a를 읽지 못함 | `ffmpeg -version`이 되는지 확인 (새 터미널) |
| Function Calling이 안 됨 | README 설명대로 `gpt-6-luna`는 `reasoning_effort="none"` 필요 |
