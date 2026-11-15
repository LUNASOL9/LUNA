"""음성 파일 → 회의록 자동 작성.

whisper_stt_last.py(Whisper STT + pyannote 화자 분리)와
summarize_and_correct.ipynb(GPT 요약·교정·문서 저장)를 하나로 이은 스크립트.

실행: python whisper\\speaker_diarization\\meeting_minutes.py
입력: 음성 파일 이름.확장자 / 화자 수 / 화자 이름
"""

import gc
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from docx import Document
from dotenv import load_dotenv
from openai import OpenAI

# 같은 폴더의 whisper_stt_last.py에 있는 설정과 함수를 그대로 가져와 쓴다
from whisper_stt_last import (
    FFMPEG_BIN,
    LANGUAGE,
    MODEL_ID,
    assign_words,
    build_final,
    save_csv,
)

AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"  # whisper/audio 폴더
GPT_MODEL = "gpt-6-luna"
CORRECTION_BATCH = 20  # GPT에게 한 번에 교정을 맡길 발화 수
CONTEXT_SIZE = 5  # 교정할 때 참고용으로 함께 보내는 앞뒤 발화 수

WORD_COLUMNS = [
    "start",
    "end",
    "speaker_id",
    "text",
    "segment_index",
    "status",
    "overlap_ratio",
    "candidate_speakers",
]
FINAL_COLUMNS = [
    "start",
    "end",
    "speaker_id",
    "duration",
    "text",
    "status",
    "review_word_count",
]


# ---------------------------------------------------------------- 입력 받기


def ask_audio_path():
    """음성 파일 이름을 입력받아 실제 경로를 돌려준다."""
    while True:
        # 탐색기의 '경로로 복사'로 붙여 넣으면 생기는 따옴표도 지운다
        name = (
            input("음성 파일 이름.확장자 (예: meeting_record.m4a): ").strip().strip('"')
        )
        if not name:
            continue
        path = Path(name)
        if not path.is_absolute():
            path = AUDIO_DIR / path  # 이름만 쓰면 whisper/audio 폴더에서 찾는다
        if path.is_file():
            return path
        print(f"  파일이 없어요: {path}")


def ask_num_speakers():
    while True:
        value = input("화자 수: ").strip()
        if value.isdigit() and int(value) > 0:
            return int(value)
        print("  1 이상의 숫자로 입력해 주세요.")


def ask_names(num_speakers):
    while True:
        value = input(f"화자 이름 (쉼표로 구분, 최대 {num_speakers}명): ")
        names = [name.strip() for name in value.split(",") if name.strip()]
        if not names:
            print("  이름을 한 명 이상 입력해 주세요.")
        elif len(names) > num_speakers:
            print(f"  이름이 화자 수({num_speakers}명)보다 많아요.")
        elif len(set(names)) != len(names):
            print("  같은 이름이 두 번 들어갔어요.")
        else:
            return names


# ---------------------------------------------------------------- 음성 처리


def load_audio(path):
    """FFmpeg로 음성 파일을 16kHz 모노 파형으로 읽는다.

    torchaudio(soundfile)는 m4a를 읽지 못해서 ffmpeg.exe로 직접 변환한다.
    mp3, m4a, wav, mp4 등 FFmpeg가 읽을 수 있는 형식은 모두 된다.
    """
    import numpy as np
    import torch

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError(
            "ffmpeg를 찾을 수 없어요. FFMPEG_BIN 경로를 확인해 주세요."
        )
    result = subprocess.run(
        # -ac 1: 모노, -ar 16000: 16kHz, -f f32le: 32비트 실수 PCM을 그대로 출력
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(path)]
        + ["-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
        check=False,  # 실패하면 아래에서 FFmpeg 오류 메시지를 직접 보여 준다
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="ignore"))
    samples = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if samples.size == 0:
        raise ValueError("오디오가 비어 있습니다.")
    return torch.from_numpy(samples).unsqueeze(0), 16000  # (채널 1개, 샘플 수)


def transcribe(waveform, sr, out):
    """Whisper로 단어별 타임스탬프가 붙은 STT 결과(chunks)를 만든다."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print("  사용 장치:", device, flush=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
    )
    generation = {"task": "transcribe"}
    if LANGUAGE:
        generation["language"] = LANGUAGE
    result = asr(
        {"raw": waveform[0].numpy(), "sampling_rate": sr},
        return_timestamps="word",
        generate_kwargs=generation,
    )
    (out / "stt_raw.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    chunks = result.get("chunks", [])
    if not chunks and result.get("text"):
        # 시간 정보가 없어도 인식 문장은 보존
        chunks = [{"timestamp": (None, None), "text": result["text"]}]

    # 화자 분리 모델을 올리기 전에 GPU 메모리를 비운다
    del asr, model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return chunks


def diarize(waveform, sr, num_speakers, token, uri, out):
    """pyannote 3.1로 화자 구간(segments)을 나눈다."""
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals(
        [TorchVersion, Specifications, Problem, Resolution]
    ):
        diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
    if diarizer is None:
        raise RuntimeError("pyannote 모델 접근 권한과 토큰을 확인해 주세요.")
    diarizer.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    annotation = diarizer(
        {"waveform": waveform, "sample_rate": sr, "uri": uri},
        num_speakers=num_speakers,
    )
    with (out / "speakers.rttm").open("w", encoding="utf-8") as f:
        annotation.write_rttm(f)
    segments = sorted(
        [
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker_id": str(speaker),
            }
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ],
        key=lambda s: (s["start"], s["end"], s["speaker_id"]),
    )
    save_csv(out / "speaker_segments.csv", segments, ["start", "end", "speaker_id"])
    return segments


# ---------------------------------------------------------------- 화자 이름 정하기


def fmt_time(seconds):
    """초를 '분:초' (1시간이 넘으면 '시:분:초') 문자열로 바꾼다."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def ask_speaker_names(rows, names):
    """화자별 대표 발화를 보여 주고, 각 화자가 누구인지 번호로 고르게 한다.

    기본값은 '아직 고르지 않은 이름 중 가장 앞의 이름'이라서, 처음 말한 순서대로
    이름을 입력했다면 Enter만 눌러도 된다.
    배경음처럼 회의록에서 뺄 화자는 0을 고른다.
    """
    # rows는 시간순이므로, 딕셔너리에 들어가는 순서가 곧 처음 말한 순서
    by_speaker = {}
    for row in rows:
        if row["speaker_id"] != "UNKNOWN" and row["text"]:
            by_speaker.setdefault(row["speaker_id"], []).append(row)

    options = "  ".join(f"{i}) {name}" for i, name in enumerate(names, start=1))
    options += "  0) 제외"
    name_map = {}
    for speaker, speaker_rows in by_speaker.items():
        used = set(name_map.values())
        unused = [i for i, name in enumerate(names, start=1) if name not in used]
        default = unused[0] if unused else 0  # 이름을 다 썼으면 기본값은 제외
        print(f"\n[{speaker}] 발화 {len(speaker_rows)}개")
        # 가장 긴 발화 3개를 시간순으로 보여 준다 (짧은 맞장구보다 누군지 알아보기 쉽다)
        longest = sorted(speaker_rows, key=lambda r: len(r["text"]), reverse=True)[:3]
        for row in sorted(longest, key=lambda r: r["start"]):
            print(f"  {fmt_time(row['start']):>7}  {row['text'][:70]}")
        while True:
            choice = input(f"누구인가요? {options}  [Enter = {default}]: ").strip()
            choice = choice or str(default)
            if choice.isdigit() and int(choice) <= len(names):
                break
            print("  목록에 있는 번호로 입력해 주세요.")
        number = int(choice)
        name_map[speaker] = names[number - 1] if number else None

    print("\n화자 이름:")
    for speaker, name in name_map.items():
        print(f"  {speaker} → {name or '(제외)'}")
    return name_map


def build_utterances(rows, name_map):
    """이름이 정해진 발화만 남기고, 같은 사람이 이어서 말한 발화는 하나로 합친다."""
    utterances = []
    for row in rows:
        name = name_map.get(row["speaker_id"])  # UNKNOWN, 제외한 화자는 None
        if not name or not row["text"]:
            continue
        if utterances and utterances[-1]["name"] == name:
            last = utterances[-1]
            last["end"] = max(last["end"], row["end"])
            last["text"] += " " + row["text"]
        else:
            utterances.append(
                {
                    "id": len(utterances) + 1,
                    "start": row["start"],
                    "end": row["end"],
                    "name": name,
                    "text": row["text"],
                }
            )
    return utterances


# ---------------------------------------------------------------- GPT 요약·교정


def to_transcript(utterances):
    """GPT에게 보낼 회의록 텍스트. 한 줄에 '[시간] 이름: 내용'."""
    return "\n".join(
        f"[{fmt_time(u['start'])}] {u['name']}: {u['text']}" for u in utterances
    )


def summarize(client, utterances):
    system_prompt = f"""
너는 회의 내용을 요약하는 봇이다. 아래 회의록을 읽고, 주요 내용을 요약하라.
결과는 마크다운 형식으로 작성한다.

# 회의 제목

## 주요 내용

## 참석자별 입장

## 결정 사항

=============== 이하 회의록 ===============
{to_transcript(utterances)}
"""
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "system", "content": system_prompt}],
    )
    return response.choices[0].message.content.strip()


def correct(client, summary, utterances):
    """STT가 잘못 받아 적은 부분을 GPT로 교정해 corrected_text에 넣는다.

    노트북처럼 발화마다 전체 녹취록을 보내면 긴 회의에서 호출 수와 비용이 크게 늘어나서,
    CORRECTION_BATCH개씩 묶어 보내고 앞뒤 CONTEXT_SIZE개 발화만 참고용으로 붙인다.
    """
    for start in range(0, len(utterances), CORRECTION_BATCH):
        end = start + CORRECTION_BATCH
        batch = utterances[start:end]
        before = utterances[max(0, start - CONTEXT_SIZE) : start]
        after = utterances[end : end + CONTEXT_SIZE]
        targets = [{"id": u["id"], "name": u["name"], "text": u["text"]} for u in batch]

        # f-string 안에서 중괄호 { }를 글자로 쓰려면 {{ }}처럼 두 번 쓴다
        system_prompt = f"""
너는 주어진 회의 녹취록을 수정 및 보완하는 봇이다.
주어진 회의 녹취록은 STT로 작성된 결과이므로, 이 중에 오류가 있는 부분을 찾아내고 수정하라.

표현, 말투는 최대한 원본과 일치하도록 유지하되, 잘못 받아적은 내용을 수정하라.
원본의 내용을 최대한 살리되, 잘못된 내용만 수정하라. 고칠 곳이 없으면 원본을 그대로 쓴다.

결과는 아래 JSON 형식으로만 작성하라. 확인할 메시지마다 id를 그대로 두고 하나씩 작성한다.
{{"corrections": [{{"id": 1, "text": "수정된 메시지"}}]}}
--------------------------------
회의 요약문: {summary}
--------------------------------
앞 내용 (참고용, 수정하지 않음):
{to_transcript(before) or "(없음)"}
--------------------------------
확인할 메시지: {json.dumps(targets, ensure_ascii=False)}
--------------------------------
뒤 내용 (참고용, 수정하지 않음):
{to_transcript(after) or "(없음)"}
"""
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "system", "content": system_prompt}],
            response_format={"type": "json_object"},  # JSON으로만 답하게 한다
        )
        try:
            items = json.loads(response.choices[0].message.content)["corrections"]
            corrected = {int(item["id"]): str(item["text"]).strip() for item in items}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            print("  교정 결과를 읽지 못해 이 묶음은 원본을 그대로 씁니다.")
            corrected = {}

        for u in batch:
            # 빠진 id나 빈 답은 원본을 그대로 쓴다
            u["corrected_text"] = corrected.get(u["id"]) or u["text"]
        print(f"  {min(end, len(utterances))}/{len(utterances)}", flush=True)


# ---------------------------------------------------------------- 저장


def add_markdown(document, text):
    """간단한 마크다운(#, ##, -)을 Word 제목과 글머리표로 옮긴다."""
    for line in text.splitlines():
        line = line.strip().replace("**", "")
        if not line:
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))  # 앞에 붙은 # 개수
            document.add_heading(line.lstrip("#").strip(), level=min(level, 3))
        elif line.startswith(("- ", "* ")):
            document.add_paragraph(line[2:], style="List Bullet")
        else:
            document.add_paragraph(line)


def save_minutes(out, audio, names, summary, utterances):
    (out / "summary.md").write_text(summary + "\n", encoding="utf-8")
    (out / "meeting_minutes.json").write_text(
        json.dumps(utterances, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown: 요약 + 녹취록
    lines = [summary, "", "---", "", "## 회의 녹취록", ""]
    lines += [
        f"- `{fmt_time(u['start'])}` **{u['name']}** : {u['corrected_text']}"
        for u in utterances
    ]
    (out / "meeting_minutes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Word: 요약 + 녹취록 표
    document = Document()
    document.add_heading("회의록", level=0)
    document.add_paragraph(f"음성 파일: {audio.name}")
    document.add_paragraph(f"참석자: {', '.join(names)}")
    add_markdown(document, summary)
    document.add_heading("회의 녹취록", level=1)
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    header = table.rows[0].cells
    header[0].text = "시간"
    header[1].text = "화자"
    header[2].text = "내용"
    for u in utterances:
        cells = table.add_row().cells
        cells[0].text = fmt_time(u["start"])
        cells[1].text = u["name"]
        cells[2].text = u["corrected_text"]
    document.save(str(out / "meeting_minutes.docx"))


# ---------------------------------------------------------------- 실행


def main():
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(".env에 HUGGING_FACE_TOKEN을 설정해 주세요.")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError(".env에 OPENAI_API_KEY를 설정해 주세요.")
    if FFMPEG_BIN.is_dir():
        os.environ["PATH"] += os.pathsep + str(FFMPEG_BIN)

    audio = ask_audio_path()
    num_speakers = ask_num_speakers()
    names = ask_names(num_speakers)

    # 매번 별도 폴더에 저장하여 이전 결과를 보존
    out = audio.parent / f"{audio.stem}_minutes_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir()

    print("\n[1/6] 오디오 읽기", flush=True)
    waveform, sr = load_audio(audio)
    print(f"  길이 {fmt_time(waveform.shape[1] / sr)}")

    print("[2/6] Whisper 단어별 STT (길이에 따라 몇 분 걸려요)", flush=True)
    chunks = transcribe(waveform, sr, out)

    print("[3/6] 화자 분리", flush=True)
    segments = diarize(waveform, sr, num_speakers, hf_token, audio.stem, out)
    words = assign_words(chunks, segments)
    review = [word for word in words if word["status"] != "matched"]
    rows = build_final(segments, words)
    save_csv(out / "words.csv", words, WORD_COLUMNS)
    save_csv(out / "review.csv", review, WORD_COLUMNS)
    save_csv(out / f"{audio.stem}_by_Whisper_stt_Final.csv", rows, FINAL_COLUMNS)
    print(f"  STT 항목 {len(words)}개 / 검토 필요 {len(review)}개")

    print("[4/6] 화자 이름 정하기", flush=True)
    name_map = ask_speaker_names(rows, names)
    utterances = build_utterances(rows, name_map)
    if not utterances:
        raise RuntimeError(
            "회의록에 넣을 발화가 없어요. 화자 이름 선택을 확인해 주세요."
        )

    client = OpenAI(api_key=openai_key)
    print(f"\n[5/6] GPT 회의 요약 (발화 {len(utterances)}개)", flush=True)
    summary = summarize(client, utterances)

    print("[6/6] GPT 발화 교정", flush=True)
    correct(client, summary, utterances)

    save_minutes(out, audio, names, summary, utterances)
    print(f"\n완료: {out}")
    print(
        "  summary.md / meeting_minutes.md / meeting_minutes.docx / meeting_minutes.json"
    )


if __name__ == "__main__":
    main()
